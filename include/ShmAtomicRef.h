#include <atomic>
#include <cstring>
#include <cassert>
#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>

template <typename Type>
class ShmAtomicRef
{
    static_assert(std::is_trivially_copyable_v<Type>, "atomic_ref<Type> requires Type to be trivially copyable.");
    static_assert(std::is_integral_v<Type> || std::is_floating_point_v<Type>, "ShmAtomicRef<Type> only supports integral or floating point types.");
    using value_type = std::remove_cv_t<Type>;
    using difference_type = value_type;
    value_type *ptr;

    long __linux_wait(value_type *addr, value_type expected)
    {
        static_assert(sizeof(Type) == 4, "Type must be 32 bit"); // For now, we only support 32-bit types due to futex limitations. Future work could explore using futex2 for 64-bit support.
        static_assert(std::is_standard_layout_v<Type>, "Type must be standard layout");
        uint32_t val_bits;
        if constexpr (std::is_same_v<value_type, uint32_t> || std::is_same_v<value_type, int32_t>)
        {
            val_bits = static_cast<uint32_t>(expected);
        }
        else
        {
            std::memcpy(&val_bits, &expected, sizeof(val_bits));
        }
        return syscall(SYS_futex, addr, FUTEX_WAIT, val_bits, nullptr, nullptr, 0);
    }

    long __linux_notify(value_type *addr, bool all = false)
    {
        int count = all ? INT_MAX : 1;
        return syscall(SYS_futex, addr, FUTEX_WAKE, count, nullptr, nullptr, 0);
    }

public:
    static constexpr bool is_always_lock_free = __atomic_always_lock_free(sizeof(value_type), 0);
    static_assert(is_always_lock_free, "atomic_ref<Type> requires Type to be always lock-free.");
    static constexpr std::size_t required_alignment = alignof(value_type);

    explicit ShmAtomicRef(Type &obj)
    {
        ptr = &obj;
    }

    ShmAtomicRef(const ShmAtomicRef &ref) noexcept
    {
        ptr = ref.ptr;
    }

    value_type operator=(value_type desired) const noexcept
    {
        __atomic_store_n(ptr, desired, std::memory_order_seq_cst);
        return desired;
    }

    ShmAtomicRef &operator=(const ShmAtomicRef &) = delete;

    bool is_lock_free() const noexcept
    {
        return __atomic_is_lock_free(sizeof(value_type), ptr);
    }

    void store(value_type desired, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        __atomic_store_n(ptr, desired, order);
    }

    value_type load(std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_acquire || order == std::memory_order_relaxed);
        return __atomic_load_n(ptr, order);
    }

    operator value_type() const noexcept
    {
        return load();
    }

    value_type exchange(value_type desired, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_exchange_n(ptr, desired, order);
    }

    bool compare_exchange_weak(value_type &expected, value_type desired, std::memory_order success, std::memory_order failure) const noexcept
    {
        assert(success == std::memory_order_seq_cst || success == std::memory_order_release || success == std::memory_order_relaxed);
        assert(failure == std::memory_order_seq_cst || failure == std::memory_order_acquire || failure == std::memory_order_relaxed);
        return __atomic_compare_exchange_n(ptr, &expected, desired, true, success, failure);
    }

    bool compare_exchange_weak(value_type &expected, value_type desired, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_compare_exchange_n(ptr, &expected, desired, true, order, order);
    }

    bool compare_exchange_strong(value_type &expected, value_type desired, std::memory_order success, std::memory_order failure) const noexcept
    {
        assert(success == std::memory_order_seq_cst || success == std::memory_order_release || success == std::memory_order_relaxed);
        assert(failure == std::memory_order_seq_cst || failure == std::memory_order_acquire || failure == std::memory_order_relaxed);
        return __atomic_compare_exchange_n(ptr, &expected, desired, false, success, failure);
    }

    bool compare_exchange_strong(value_type &expected, value_type desired, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_compare_exchange_n(ptr, &expected, desired, false, order, order);
    }

    void wait(value_type old, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_acquire || order == std::memory_order_relaxed);
        __linux_wait(ptr, old);
    }

    void notify_one() const noexcept
    {
        __linux_notify(ptr, false);
    }

    void notify_all() const noexcept
    {
        __linux_notify(ptr, true);
    }

    constexpr Type *address() const noexcept
    {
        return ptr;
    }

    value_type fetch_add(difference_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_add(ptr, arg, (int)order);
    }

    value_type fetch_sub(difference_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_sub(ptr, arg, (int)order);
    }

    value_type operator+=(difference_type arg) const noexcept
    {
        return fetch_add(arg) + arg;
    }

    value_type operator-=(difference_type arg) const noexcept
    {
        return fetch_sub(arg) - arg;
    }

    value_type fetch_and(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_and(ptr, arg, (int)order);
    }

    value_type fetch_or(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_or(ptr, arg, (int)order);
    }

    value_type fetch_xor(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_xor(ptr, arg, (int)order);
    }

    value_type fetch_nand(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);
        return __atomic_fetch_nand(ptr, arg, (int)order);
    }

    value_type operator&=(value_type arg) const noexcept
    {
        return fetch_and(arg) & arg;
    }

    value_type operator|=(value_type arg) const noexcept
    {
        return fetch_or(arg) | arg;
    }

    value_type operator^=(value_type arg) const noexcept
    {
        return fetch_xor(arg) ^ arg;
    }

    value_type fetch_max(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);

        value_type prev = load(std::memory_order_relaxed);
        while (prev < arg && !compare_exchange_weak(prev, arg, order))
        {
        }
        return prev;
    }

    value_type fetch_min(value_type arg, std::memory_order order = std::memory_order_seq_cst) const noexcept
    {
        assert(order == std::memory_order_seq_cst || order == std::memory_order_release || order == std::memory_order_relaxed);

        value_type prev = load(std::memory_order_relaxed);
        while (prev > arg && !compare_exchange_weak(prev, arg, order))
        {
        }
        return prev;
    }

    value_type operator++() const noexcept
    {
        return fetch_add(1) + 1;
    }

    value_type operator++(int) const noexcept
    {
        return fetch_add(1);
    }

    value_type operator--() const noexcept
    {
        return fetch_sub(1) - 1;
    }

    value_type operator--(int) const noexcept
    {
        return fetch_sub(1);
    }
};
