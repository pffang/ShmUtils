#pragma once

#include <pthread.h>
#include <cassert>
#include <system_error>
#include <condition_variable>
#include "ShmMutex.h"

class posix_cond_attr
{
    pthread_condattr_t m_attr;

public:
    posix_cond_attr()
    {
        if(pthread_condattr_init(&m_attr) != 0)
            throw std::system_error(errno, std::system_category(), "pthread_condattr_init failed");
        if(pthread_condattr_setpshared(&m_attr, PTHREAD_PROCESS_SHARED) != 0)
            throw std::system_error(errno, std::system_category(), "pthread_condattr_setpshared failed");
    }

    ~posix_cond_attr() { pthread_condattr_destroy(&m_attr); }

    operator pthread_condattr_t&() { return m_attr; }
};

class condition_initializer
{
    pthread_cond_t* mp_cond = nullptr;

public:
    condition_initializer(pthread_cond_t& cond, pthread_condattr_t& cond_attr) : mp_cond(&cond)
    {
        if(pthread_cond_init(mp_cond, &cond_attr) != 0)
            throw std::system_error(errno, std::system_category(), "pthread_cond_init failed");
    }

    ~condition_initializer()
    {
        if(mp_cond)
            pthread_cond_destroy(mp_cond);
    }

    void release() { mp_cond = nullptr; }
};

class ShmCondition
{
    pthread_cond_t m_cond = PTHREAD_COND_INITIALIZER;

    ShmCondition(const ShmCondition&)            = delete;
    ShmCondition& operator=(const ShmCondition&) = delete;

public:
    ShmCondition()
    {
        posix_cond_attr       cond_attr;
        condition_initializer cond(m_cond, cond_attr);
        cond.release();
    }

    ~ShmCondition()
    {
        int res = pthread_cond_destroy(&m_cond);
        assert(res == 0);
        (void)res;
    }

    void notify_one() noexcept
    {
        int res = pthread_cond_signal(&m_cond);
        (void)res;
    }

    void notify_all() noexcept
    {
        int res = pthread_cond_broadcast(&m_cond);
        (void)res;
    }

    template <class Lock>
    void wait(Lock& lock)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        int res = pthread_cond_wait(&m_cond, lock.mutex()->native_handle());
        if(res != 0)
            throw std::system_error(res, std::system_category(), "pthread_cond_wait failed");
    }

    template <class Lock, class Predicate>
    void wait(Lock& lock, Predicate pred)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        while(!pred())
            wait(lock);
    }

    template <class Lock, class Rep, class Period>
    std::cv_status wait_for(Lock&                                     lock,
                            const std::chrono::duration<Rep, Period>& rel_time)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait_for only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        const auto rel = std::chrono::duration_cast<std::chrono::nanoseconds>(rel_time);
        timespec   ts  = monotonic_deadline_from_now(rel);
        int      res = pthread_cond_clockwait(&m_cond, lock.mutex()->native_handle(), CLOCK_MONOTONIC, &ts);
        if(res == 0)
            return std::cv_status::no_timeout;
        else if(res == ETIMEDOUT)
            return std::cv_status::timeout;
        else
            throw std::system_error(res, std::system_category(), "pthread_cond_clockwait failed");
    }

    template <class Lock, class Rep, class Period, class Predicate>
    bool wait_for(Lock&                                     lock,
                  const std::chrono::duration<Rep, Period>& rel_time,
                  Predicate                                 pred)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait_for only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        while(!pred())
        {
            if(wait_for(lock, rel_time) == std::cv_status::timeout)
                return false;
        }
        return true;
    }

    template <class Lock, class Clock, class Duration>
    std::cv_status
    wait_until(Lock&                                           lock,
               const std::chrono::time_point<Clock, Duration>& abs_time)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait_until only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        timespec ts  = timepoint_to_timespec(abs_time);
        int      res = pthread_cond_timedwait(&m_cond, lock.mutex()->native_handle(), &ts);
        if(res == 0)
            return std::cv_status::no_timeout;
        else if(res == ETIMEDOUT)
            return std::cv_status::timeout;
        else
            throw std::system_error(res, std::system_category(), "pthread_cond_timedwait failed");
    }

    template <class Lock, class Clock, class Duration, class Predicate>
    bool wait_until(Lock&                                           lock,
                    const std::chrono::time_point<Clock, Duration>& abs_time,
                    Predicate                                       pred)
    {
        static_assert(std::is_same<typename Lock::mutex_type, ShmMutex>::value || std::is_same<typename Lock::mutex_type, ShmMutexRecursive>::value,
                      "ShmCondition::wait_until only supports Lock<ShmMutex> or Lock<ShmMutexRecursive>");
        while(!pred())
        {
            if(wait_until(lock, abs_time) == std::cv_status::timeout)
                return false;
        }
        return true;
    }
};