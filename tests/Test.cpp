#include "Common.h"

#include <iostream>
#include <thread>
#include <vector>

volatile int counter(0); // 非原子计数器
ShmMutex mtx;            // 锁定对计数器的访问

void attempt_10k_increases()
{
    for (int i = 0; i < 10000; ++i)
    {
        if (mtx.try_lock())
        { // 只有在未锁定时才增加：
            ++counter;
            mtx.unlock();
        }
    }
}

int Test1ShmMutexLockUnlock()
{
    std::thread threads[10];
    for (int i = 0; i < 10; ++i)
        threads[i] = std::thread(attempt_10k_increases);
    for (auto &th : threads)
        th.join();
    std::cout << counter << " successful increases of the counter.\n";
    return 0;
}

void print_even(int x)
{
    if (x % 2 == 0)
        std::cout << x << " is even\n";
    else
        throw(std::logic_error("not even"));
}

void print_thread_id(int id)
{
    try
    {
        // 使用局部的lock_guard来锁定mtx，保证在析构/异常时解锁：
        std::lock_guard<ShmMutex> lck(mtx);
        print_even(id);
    }
    catch (std::logic_error &)
    {
        std::cout << "[exception caught]\n";
    }
}

int Test2ShmMutexLockGuard()
{
    std::thread threads[10];
    for (int i = 0; i < 10; ++i)
        threads[i] = std::thread(print_thread_id, i + 1);
    for (auto &th : threads)
        th.join();
    return 0;
}

ShmCondition cv;
ShmMutex cv_m; // This mutex is used for three purposes:
               // 1) to synchronize accesses to i
               // 2) to synchronize accesses to std::cerr
               // 3) for the condition variable cv
int i = 0;

void waits()
{
    std::unique_lock<ShmMutex> lk(cv_m);
    std::cerr << "Waiting... \n";
    cv.wait(lk, []
            { return i == 1; });
    std::cerr << "...finished waiting. i == 1\n";
}

void signals()
{
    std::this_thread::sleep_for(std::chrono::seconds(1));
    {
        std::lock_guard<ShmMutex> lk(cv_m);
        std::cerr << "Notifying...\n";
    }
    cv.notify_all();

    std::this_thread::sleep_for(std::chrono::seconds(1));

    {
        std::lock_guard<ShmMutex> lk(cv_m);
        i = 1;
        std::cerr << "Notifying again...\n";
    }
    cv.notify_all();
}

void Test3ShmConditionWait()
{
    std::thread t1(waits), t2(waits), t3(waits), t4(signals);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
}

int acnt;
int cnt;

void f()
{
    ShmAtomicRef<int> acnt_ref(acnt);
    for (auto n{10000}; n; --n)
    {
        ++acnt_ref;
        ++cnt;
        // Note: for this example, relaxed memory order is sufficient,
        // e.g. acnt.fetch_add(1, std::memory_order_relaxed);
    }
}

void TestShmAtomicRef()
{
    {
        std::vector<std::jthread> pool;
        for (int n = 0; n < 10; ++n)
            pool.emplace_back(f);
    }

    std::cout << "The atomic counter is " << acnt << '\n'
              << "The non-atomic counter is " << cnt << '\n';
}

int main(int argc, const char *argv[])
{
    Test1ShmMutexLockUnlock();
    Test2ShmMutexLockGuard();
    Test3ShmConditionWait();
    TestShmAtomicRef();

    return 0;
}