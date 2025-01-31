#pragma once

#include <condition_variable>
#include <functional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

/**
 * @brief A ThreadPool for tasks with signature: void(T...).
 *
 * It executes a single user-supplied "main function" for each set of arguments
 * you enqueue, plus optional "onStart" / "onFinish" callbacks (also taking T...).
 *
 * The pool can either spawn threads immediately (run_immediately = true),
 * or wait until the first call to wait() (run_immediately = false).
 *
 * Usage Example:
 *   auto pool = ThreadPool<std::string,int,int>(
 *       4,
 *       [](std::string s, int x, int y){ },
 *       true,  // run_immediately
 *       [](auto&&... args) {},
 *       [](auto&&... args) {}
 *   );
 *
 *   // Enqueue tasks:
 *   pool("task1", 10, 12);
 *   pool("task2", 23, 42);
 *
 *   // Wait until all tasks complete:
 *   pool.wait();
 */

// Main Task Pool
template <typename... Args>
class task_pool {
public:
	/// The main function to run for each enqueued argument set.
	using Task = std::function<void(Args...)>;

	/// Optional callbacks that can also receive the same (T...) arguments.
	using Callback = std::function<void(const Args&...)>;

	/**
	 * @brief Construct the ThreadPool.
	 *
	 * @param num_threads        Number of worker threads to spawn.
	 * @param main_func          Function invoked for each enqueued (T...) set.
	 * @param run_immediately   If true, spawn threads in the constructor.
	 *                          If false, threads are spawned only upon the first wait() call.
	 * @param on_start           Callback before each task (may be empty).
	 * @param on_finish          Callback after each task (may be empty).
	 */
	explicit task_pool(Task main_func, const std::size_t num_threads = std::thread::hardware_concurrency(), const bool run_immediately = true,
		Callback on_start = {}, Callback on_finish = {}) :
		main_func_(std::move(main_func)), on_start_(std::move(on_start)), on_finish_(std::move(on_finish)), stop_(false),
		started_(false), num_threads_(num_threads), total_queued_(0), total_finished_(0), tasks_in_queue_(0),
		tasks_in_progress_(0) {
		if (!main_func_) {
			throw std::invalid_argument("ThreadPool: main_func cannot be empty");
		}
		if (num_threads_ == 0) {
			throw std::invalid_argument("ThreadPool: num_threads must be > 0");
		}

		// If user wants immediate start, spawn threads now
		if (run_immediately) {
			start_threads();
		}
	}

	/**
	 * @brief Destructor - stops the pool and joins all threads.
	 */
	~task_pool() { stop(); }

	/**
	 * @brief Enqueue a new task by calling operator()(args...).
	 *
	 * This will store the arguments to be passed to the main function.
	 * The actual function call happens on a worker thread (once running).
	 */
	void operator()(Args... args) { enqueue(std::forward<Args>(args)...); }

	/**
	 * @brief Wait until all currently queued tasks (and any in progress) are finished.
	 *
	 * If the pool wasn't started yet (run_immediately == false), this will spawn
	 * the threads on the first call.
	 */
	void wait() {
		{
			// If not started (and user asked for lazy start), do so now
			std::unique_lock<std::mutex> lock(mutex_);
			if (!started_) {
				start_threads();
			}
		}

		// Now block until everything finishes
		std::unique_lock<std::mutex> lock(mutex_);
		wait_cond_var_.wait(lock, [this] { return (tasks_in_queue_ == 0 && tasks_in_progress_ == 0); });
	}

	/**
	 * @brief Returns total number of tasks ever enqueued since creation.
	 */
	int get_total() {
		std::unique_lock<std::mutex> lock(mutex_);
		return total_queued_;
	}

	/**
	 * @brief Returns total number of tasks that have finished.
	 */
	int get_finished() {
		std::unique_lock<std::mutex> lock(mutex_);
		return total_finished_;
	}

	/**
	 * @brief Returns the count of tasks not yet finished (still in queue or in progress).
	 */
	int get_remaining() {
		std::unique_lock<std::mutex> lock(mutex_);
		// Alternatively: (totalQueued_ - totalFinished_), but let's keep the separate counters
		return static_cast<int>(tasks_in_queue_ + tasks_in_progress_);
	}

	// Non-copyable
	task_pool(const task_pool&) = delete;
	task_pool& operator=(const task_pool&) = delete;

private:
	/**
	 * @brief Enqueue the given arguments into the task queue.
	 */
	void enqueue(Args... args) {
		std::unique_lock<std::mutex> lock(mutex_);
		if (stop_) {
			throw std::runtime_error("ThreadPool: cannot enqueue on a stopped pool");
		}
		// Package the arguments into a tuple
		task_queue_.emplace(std::make_tuple(std::forward<Args>(args)...));
		tasks_in_queue_++;
		total_queued_++;
		// Wake one waiting worker thread (if already started)
		cond_var_.notify_one();
	}

	/**
	 * @brief Actually spawn worker threads (called once).
	 */
	void start_threads() {
		if (started_) {
			return; // Already started
		}
		started_ = true; // Mark started

		workers_.reserve(num_threads_);
		for (std::size_t i = 0; i < num_threads_; ++i) {
			workers_.emplace_back([this] { worker_loop(); });
		}
	}

	/**
	 * @brief The main worker loop, executed by each thread.
	 */
	void worker_loop() {
		for (;;) {
			{
				std::unique_lock<std::mutex> lock(mutex_);
				// Wait until there's a task or we're stopping
				cond_var_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });

				// If stopping and no tasks left, exit the loop (end thread)
				if (stop_ && task_queue_.empty()) {
					return;
				}

				// Dequeue the next task
				std::tuple<Args...> task_args = std::move(task_queue_.front());

				task_queue_.pop();
				tasks_in_queue_--;
				tasks_in_progress_++;

				if (on_start_) {
					std::apply(on_start_, task_args);
				}

				lock.unlock();  // Unlock before execution
				try {
					std::apply(main_func_, task_args);
				} catch (...) {}

				lock.lock();

				if (on_finish_) {
					std::apply(on_finish_, task_args);
				}

				tasks_in_progress_--;
				total_finished_++;
				if (tasks_in_queue_ == 0 && tasks_in_progress_ == 0) {
					wait_cond_var_.notify_all();
				}
			}
		}
	}

	/**
	 * @brief Stop the pool, discard remaining tasks, and join all threads.
	 */
	void stop() {
		{
			std::unique_lock<std::mutex> lock(mutex_);
			if (!stop_) {
				stop_ = true;
				// Clear any remaining tasks that haven't started
				while (!task_queue_.empty()) {
					task_queue_.pop();
					tasks_in_queue_--;
				}
			}
		}
		cond_var_.notify_all();

		// Join threads
		for (auto& thr: workers_) {
			if (thr.joinable()) {
				thr.join();
			}
		}
		workers_.clear();
	}

private:
	// Main function to invoke for each enqueued tuple of (T...)
	Task main_func_;

	// Optional callbacks; each can see the arguments (const T&...)
	Callback on_start_;
	Callback on_finish_;

	// Whether we've told threads to stop
	bool stop_;

	// Whether threads have actually been spawned
	bool started_;

	// Desired number of worker threads
	std::size_t num_threads_;

	// Thread list
	std::vector<std::thread> workers_;

	// Queue of tasks; each is the argument tuple for mainFunc_
	std::queue<std::tuple<Args...>> task_queue_;

	// Synchronization
	std::mutex mutex_;
	std::condition_variable cond_var_;
	std::condition_variable wait_cond_var_;

	// Counters
	int total_queued_; // total tasks ever enqueued
	int total_finished_; // total tasks that have completed
	std::size_t tasks_in_queue_; // how many tasks are waiting
	std::size_t tasks_in_progress_; // how many tasks are currently running
};