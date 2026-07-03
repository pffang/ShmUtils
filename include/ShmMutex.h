#pragma once

#include <pthread.h>
#include <cassert>
#include <ctime>
#include <system_error>
#include <chrono>
#include <stdexcept>
#include <mutex>

inline constexpr timespec duration_to_timespec(const std::chrono::nanoseconds& ns)
{
    auto secs  = std::chrono::duration_cast<std::chrono::seconds>(ns);
    auto nanos = ns - secs;
    return {secs.count(), nanos.count()};
}

inline timespec monotonic_deadline_from_now(const std::chrono::nanoseconds& rel) noexcept
{
    timespec now {};
    (void)::clock_gettime(CLOCK_MONOTONIC, &now);

    const auto now_ns = std::chrono::seconds(now.tv_sec) + std::chrono::nanoseconds(now.tv_nsec);
    const auto end_ns = now_ns + rel;
    return duration_to_timespec(end_ns);
}

inline constexpr timespec timepoint_to_timespec(std::chrono::system_clock::time_point tp)
{
    auto d = tp.time_since_epoch();
    return duration_to_timespec(std::chrono::duration_cast<std::chrono::nanoseconds>(d));
}

class posix_mutex_attr
{
    pthread_mutexattr_t m_attr;

public:
    posix_mutex_attr(bool recursive = false)
    {
        int res = pthread_mutexattr_init(&m_attr);
        if(res != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutexattr_init failed");

        if((res = pthread_mutexattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED)) != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutexattr_setpshared failed");
        if((res = pthread_mutexattr_setrobust(&m_attr, PTHREAD_MUTEX_ROBUST)) != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutexattr_setrobust failed");

        if(recursive && (res = pthread_mutexattr_settype(&m_attr, PTHREAD_MUTEX_RECURSIVE)) != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutexattr_settype failed");
    }

    ~posix_mutex_attr() { pthread_mutexattr_destroy(&m_attr); }

    operator pthread_mutexattr_t&() { return m_attr; }
};

class posix_mutex_initializer
{
    pthread_mutex_t* mp_mut = nullptr;

public:
    posix_mutex_initializer(pthread_mutex_t& mut, pthread_mutexattr_t& mut_attr) : mp_mut(&mut)
    {
        int res = pthread_mutex_init(mp_mut, &mut_attr);
        if(res != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutex_init failed");
    }

    ~posix_mutex_initializer()
    {
        if(mp_mut)
            pthread_mutex_destroy(mp_mut);
    }

    void release() { mp_mut = nullptr; }
};

class posix_mutex
{
    pthread_mutex_t m_mutex = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP;

    posix_mutex(const posix_mutex&)            = delete;
    posix_mutex& operator=(const posix_mutex&) = delete;

public:
    posix_mutex(bool recursive = false)
    {
        posix_mutex_attr        mut_attr(recursive);
        posix_mutex_initializer mut(m_mutex, mut_attr);
        mut.release();
    }

    ~posix_mutex()
    {
        int res = pthread_mutex_destroy(&m_mutex);
        assert(res == 0);
        (void)res;
    }

    pthread_mutex_t* native_handle() { return &m_mutex; }

private:
    void make_consistent_or_throw(const char* api_name)
    {
        int consistent_res = pthread_mutex_consistent(&m_mutex);
        if(consistent_res != 0)
            throw std::system_error(consistent_res, std::system_category(), api_name);
    }

public:

    void lock()
    {
        int res = pthread_mutex_lock(&m_mutex);
        if(res == EOWNERDEAD)
        {
            make_consistent_or_throw("pthread_mutex_lock(pthread_mutex_consistent) failed");
            return;
        }
        else if(res == ENOTRECOVERABLE)
            throw std::system_error(ENOTRECOVERABLE, std::system_category(), "pthread_mutex_lock failed");
        else if(res != 0)
            throw std::system_error(res, std::system_category(), "pthread_mutex_lock failed");
    }

    bool try_lock()
    {
        int res = pthread_mutex_trylock(&m_mutex);

        if(res == EOWNERDEAD)
        {
            make_consistent_or_throw("pthread_mutex_trylock(pthread_mutex_consistent) failed");
            return true;
        }
        else if(res == ENOTRECOVERABLE)
            throw std::system_error(ENOTRECOVERABLE, std::system_category(), "pthread_mutex_trylock failed");
        else if(!(res == 0 || res == EBUSY))
            throw std::system_error(res, std::system_category(), "pthread_mutex_trylock failed");
        return (res == 0);
    }

    template <class Rep, class Period>
    bool try_lock_for(const std::chrono::duration<Rep, Period>& timeout_duration)
    {
        const auto rel = std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_duration);
        timespec   ts  = monotonic_deadline_from_now(rel);
        int      res = pthread_mutex_clocklock(&m_mutex, CLOCK_MONOTONIC, &ts);

        if(res == EOWNERDEAD)
        {
            make_consistent_or_throw("pthread_mutex_clocklock(pthread_mutex_consistent) failed");
            return true;
        }
        else if(res == ENOTRECOVERABLE)
            throw std::system_error(ENOTRECOVERABLE, std::system_category(), "pthread_mutex_clocklock failed");
        else if(res != 0 && res != ETIMEDOUT)
            throw std::system_error(res, std::system_category(), "pthread_mutex_clocklock failed");
        return res == 0;
    }

    template <class Clock, class Duration>
    bool try_lock_until(const std::chrono::time_point<Clock, Duration>& timeout_time)
    {
        timespec ts  = timepoint_to_timespec(timeout_time);
        int      res = pthread_mutex_timedlock(&m_mutex, &ts);
        if(res == EOWNERDEAD)
        {
            make_consistent_or_throw("pthread_mutex_timedlock(pthread_mutex_consistent) failed");
            return true;
        }
        else if(res == ENOTRECOVERABLE)
            throw std::system_error(ENOTRECOVERABLE, std::system_category(), "pthread_mutex_timedlock failed");
        else if(res != 0 && res != ETIMEDOUT)
            throw std::system_error(res, std::system_category(), "pthread_mutex_timedlock failed");
        return res == 0;
    }

    void unlock()
    {
        int res = 0;
        res     = pthread_mutex_unlock(&m_mutex);
        assert(res == 0);
        (void)res;
    }
};

class ShmMutex : public posix_mutex
{
    ShmMutex(const ShmMutex&)            = delete;
    ShmMutex& operator=(const ShmMutex&) = delete;

public:
    ShmMutex() : posix_mutex(false)
    {
    }

    ~ShmMutex() = default;
};

class ShmMutexRecursive : public posix_mutex
{
    ShmMutexRecursive(const ShmMutexRecursive&)   = delete;
    ShmMutexRecursive& operator=(const ShmMutex&) = delete;

public:
    ShmMutexRecursive() : posix_mutex(true)
    {
    }

    ~ShmMutexRecursive() = default;
};