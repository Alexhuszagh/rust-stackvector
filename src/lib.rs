//! Vector-like class allocated entirely on the stack.
//!
//! Shallow wrapper around an underlying `Array`, which panics if the
//! array bounds are exceeded.
//!
//! # no_std support
//!
//! By default, `smallvec` depends on `libstd`. However, it can be configured to use the unstable
//! `liballoc` API instead, for use on platforms that have `liballoc` but not `libstd`.  This
//! configuration is currently unstable and is not guaranteed to work on all versions of Rust.
//!
//! To depend on `smallvec` without `libstd`, use `default-features = false` in the `smallvec`
//! section of Cargo.toml to disable its `"std"` feature.
//!
//! Adapted from Servo's smallvec:
//!     https://github.com/servo/rust-smallve
//!
//! StackVec is distributed under the same terms as the smallvec and
//! lexical, that is, it is dual licensed under either the MIT or Apache
//! 2.0 license.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::{Borrow, BorrowMut};
use alloc::vec::Vec;
use core::{cmp, fmt, hash, iter, mem, ops, ptr, slice};

// VEC LIKE

/// Common operations implemented by both `Vec` and `StackVec`.
///
/// This can be used to write generic code that works with both `Vec` and `StackVec`.
///
/// ## Example
///
/// ```rust
/// use stackvector::{VecLike, StackVec};
///
/// fn initialize<V: VecLike<u8>>(v: &mut V) {
///     for i in 0..5 {
///         v.push(i);
///     }
/// }
///
/// let mut vec = Vec::new();
/// initialize(&mut vec);
///
/// let mut stack_vec = StackVec::<u8, 8>::new();
/// initialize(&mut stack_vec);
/// ```
pub trait VecLike<T>:
    ops::Index<usize, Output = T>
    + ops::IndexMut<usize>
    + ops::Index<ops::Range<usize>, Output = [T]>
    + ops::IndexMut<ops::Range<usize>>
    + ops::Index<ops::RangeFrom<usize>, Output = [T]>
    + ops::IndexMut<ops::RangeFrom<usize>>
    + ops::Index<ops::RangeTo<usize>, Output = [T]>
    + ops::IndexMut<ops::RangeTo<usize>>
    + ops::Index<ops::RangeFull, Output = [T]>
    + ops::IndexMut<ops::RangeFull>
    + ops::DerefMut<Target = [T]>
    + Extend<T>
{
    /// Append an element to the vector.
    fn push(&mut self, value: T);

    /// Pop an element from the end of the vector.
    fn pop(&mut self) -> Option<T>;
}

#[allow(deprecated)]
impl<T> VecLike<T> for Vec<T> {
    #[inline]
    fn push(&mut self, value: T) {
        Vec::push(self, value);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        Vec::pop(self)
    }
}

// EXTEND FROM SLICE

/// Trait to be implemented by a collection that can be extended from a slice
///
/// ## Example
///
/// ```rust
/// use stackvector::{ExtendFromSlice, StackVec};
///
/// fn initialize<V: ExtendFromSlice<u8>>(v: &mut V) {
///     v.extend_from_slice(b"Test!");
/// }
///
/// let mut vec = Vec::new();
/// initialize(&mut vec);
/// assert_eq!(&vec, b"Test!");
///
/// let mut stack_vec = StackVec::<u8, 8>::new();
/// initialize(&mut stack_vec);
/// assert_eq!(&stack_vec as &[_], b"Test!");
/// ```
pub trait ExtendFromSlice<T> {
    /// Extends a collection from a slice of its element type
    fn extend_from_slice(&mut self, other: &[T]);
}

impl<T: Clone> ExtendFromSlice<T> for Vec<T> {
    fn extend_from_slice(&mut self, other: &[T]) {
        Vec::extend_from_slice(self, other)
    }
}

// DRAIN

/// An iterator that removes the items from a `StackVec` and yields them by value.
///
/// Returned from [`StackVec::drain`][1].
///
/// [1]: struct.StackVec.html#method.drain
pub struct Drain<'a, T: 'a> {
    iter: slice::IterMut<'a, T>,
}

impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter
            .next()
            .map(|reference| unsafe { ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter
            .next_back()
            .map(|reference| unsafe { ptr::read(reference) })
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T: 'a> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}
    }
}

// SET LEN ON DROP

/// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
///
/// Copied from https://github.com/rust-lang/rust/pull/36355
struct SetLenOnDrop<'a> {
    len: &'a mut usize,
    local_len: usize,
}

impl<'a> SetLenOnDrop<'a> {
    #[inline]
    fn new(len: &'a mut usize) -> Self {
        SetLenOnDrop {
            local_len: *len,
            len,
        }
    }

    #[inline]
    unsafe fn increment_len(&mut self, n: usize) {
        self.local_len += n;
    }

    #[inline]
    unsafe fn decrement_len(&mut self, n: usize) {
        self.local_len -= n;
    }
}

impl<'a> Drop for SetLenOnDrop<'a> {
    #[inline]
    fn drop(&mut self) {
        *self.len = self.local_len;
    }
}

struct DropOnPanic<T> {
    start: *mut T,
    skip: ops::Range<usize>,
    len: usize,
}

impl<T> Drop for DropOnPanic<T> {
    fn drop(&mut self) {
        for i in 0..self.len {
            if !self.skip.contains(&i) {
                unsafe {
                    ptr::drop_in_place(self.start.add(i));
                }
            }
        }
    }
}

// STACKVEC

/// A `Vec`-like container that stores elements on the stack.
///
/// The amount of data that a `StackVec` can store inline depends on its backing store. The backing
/// store can be any type that implements the `Array` trait; usually it is a small fixed-sized
/// array.  For example a `StackVec<u64, 8>` can hold up to eight 64-bit integers inline.
///
/// ## Example
///
/// ```rust,should_panic
/// use stackvector::StackVec;
/// let mut v = StackVec::<u8, 4>::new(); // initialize an empty vector
///
/// // The vector can hold up to 4 items without spilling onto the heap.
/// v.extend(0..4);
/// assert_eq!(v.len(), 4);
///
/// // Pushing another element will force the buffer to spill and panic:
/// v.push(4);
/// ```
pub struct StackVec<T, const N: usize> {
    // The capacity field is used for iteration and other optimizations.
    // Publicly expose the fields, so they may be used in constant
    // initialization.
    data: [mem::MaybeUninit<T>; N],
    length: usize,
}

impl<T, const N: usize> StackVec<T, N> {
    /// Construct an empty vector
    #[inline]
    pub fn new() -> StackVec<T, N> {
        StackVec {
            length: 0,
            data: [const { mem::MaybeUninit::uninit() }; N],
        }
    }

    /// Construct a new `StackVec` from a `Vec<T>`.
    ///
    /// Elements will be copied to the inline buffer if vec.len() <= N.
    ///
    /// ```rust
    /// # #[cfg(feature = "std")] {
    ///
    /// extern crate alloc;
    ///
    /// use alloc::vec;
    ///
    /// use stackvector::StackVec;
    ///
    /// let vec = vec![1, 2, 3, 4, 5];
    /// let stack_vec: StackVec<_, 5> = StackVec::from_vec(vec);
    ///
    /// assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
    /// # }
    /// ```
    #[inline]
    pub fn from_vec(vec: Vec<T>) -> StackVec<T, N> {
        assert!(vec.len() <= N);
        unsafe { Self::from_vec_unchecked(vec) }
    }

    /// Construct a new `StackVec` from a `Vec<T>` without bounds checking.
    #[allow(deprecated)]
    pub unsafe fn from_vec_unchecked(vec: Vec<T>) -> StackVec<T, N> {
        let mut v = Self::new();
        let len = vec.len();
        for (index, item) in vec.into_iter().enumerate() {
            v.data[index].write(item);
        }
        v.length = len;

        v
    }

    /// Constructs a new `StackVec` on the stack from an `A` without
    /// copying elements.
    ///
    /// ```rust
    /// use stackvector::StackVec;
    ///
    /// let buf = [1, 2, 3, 4, 5];
    /// let stack_vec: StackVec<i32, 5> = StackVec::from_buf(buf);
    ///
    /// assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_buf(buf: [T; N]) -> StackVec<T, N> {
        let len = buf.len();
        unsafe { StackVec::from_buf_and_len_unchecked(buf, len) }
    }

    /// Constructs a new `StackVec` on the stack from an `[T; N]` without
    /// copying elements. Also sets the length, which must be less or
    /// equal to the size of `buf`.
    ///
    /// ```rust
    /// use stackvector::StackVec;
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let stack_vec: StackVec<i32, 8> = StackVec::from_buf_and_len(buf, 5);
    ///
    /// assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub fn from_buf_and_len(buf: [T; N], len: usize) -> StackVec<T, N> {
        assert!(len <= N && len <= buf.len());
        unsafe { StackVec::from_buf_and_len_unchecked(buf, len) }
    }

    /// Constructs a new `StackVec` on the stack from an `A` without
    /// copying elements. Also sets the length. The user is responsible
    /// for ensuring that `len <= N`.
    ///
    /// ```rust
    /// use stackvector::StackVec;
    ///
    /// let buf = [1, 2, 3, 4, 5, 0, 0, 0];
    /// let stack_vec: StackVec<i32, 8> = unsafe {
    ///     StackVec::from_buf_and_len_unchecked(buf, 5)
    /// };
    ///
    /// assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
    /// ```
    #[inline]
    pub unsafe fn from_buf_and_len_unchecked(buf: [T; N], len: usize) -> StackVec<T, N> {
        let mut v = Self::new();
        {
            let mut local_len = SetLenOnDrop::new(&mut v.length);
            for (index, item) in buf.into_iter().take(len).enumerate() {
                v.data[index].write(item);
                unsafe { local_len.increment_len(1) };
            }
        }

        v
    }

    /// Sets the length of a vector.
    ///
    /// This will explicitly set the size of the vector, without actually
    /// modifying its buffers, so it is up to the caller to ensure that the
    /// vector is actually the specified size.
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= N);
        self.length = new_len;
    }

    /// The number of elements stored in the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    /// If the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of items the vector can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Empty the vector and return an iterator over its former contents.
    pub fn drain(&mut self) -> Drain<'_, T> {
        unsafe {
            let slice = slice::from_raw_parts_mut(self.as_mut_ptr(), self.len());
            // NOTE: Cannot be `set_len` due to stack borrow rules
            self.length = 0;

            Drain {
                iter: slice.iter_mut(),
            }
        }
    }

    /// Append an item to the vector.
    #[inline]
    pub fn push(&mut self, value: T) {
        assert!(self.len() < self.capacity());
        unsafe {
            let len = self.len();
            self.data[len].write(value);
            self.set_len(len + 1);
        }
    }

    /// Remove an item from the end of the vector and return it, or None if empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        let len = self.len();
        if len == 0 {
            None
        } else {
            unsafe {
                self.set_len(len - 1);
                let init = self.as_ptr().add(self.len());
                Some(ptr::read(init))
            }
        }
    }

    /// Shorten the vector, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than or equal to the vector's current length, this has no
    /// effect.
    /// `shrink_to_fit` after truncating.
    pub fn truncate(&mut self, len: usize) {
        unsafe {
            while len < self.len() {
                self.set_len(self.len() - 1);
                self.data[self.len()].assume_init_drop();
            }
        }
    }

    /// Returns a raw pointer to the slice’s buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns,
    /// or else it will end up dangling.
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }

    /// Returns an unsafe mutable pointer to the slice’s buffer.
    ///
    /// The caller must ensure that the slice outlives the pointer this function returns,
    /// or else it will end up dangling.
    fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr() as *mut T
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Remove the element at position `index`, replacing it with the last element.
    ///
    /// This does not preserve ordering, but is O(1).
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        let len = self.len();
        self.swap(len - 1, index);
        unsafe { self.pop().unwrap_unchecked() }
    }

    /// Remove all elements from the vector.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Remove and return the element at position `index`, shifting all elements after it to the
    /// left.
    ///
    /// Panics if `index` is out of bounds.
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len());
        unsafe {
            self.length -= 1;
            let ptr = self.as_mut_ptr().add(index);
            let item = ptr::read(ptr);
            ptr::copy(ptr.offset(1), ptr, self.length - index);
            item
        }
    }

    /// Insert an element at position `index`, shifting all elements after it to the right.
    ///
    /// Panics if `index` is out of bounds.
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index < self.len() && self.len() < self.capacity());
        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            ptr::copy(ptr, ptr.offset(1), self.length - index);
            ptr::write(ptr, element);
            self.length += 1;
        }
    }

    /// Insert multiple elements at position `index`, shifting all following elements toward the
    /// back.
    pub fn insert_many<I: iter::IntoIterator<Item = T>>(&mut self, index: usize, iterable: I) {
        let mut iter = iterable.into_iter();
        if index == self.len() {
            return self.extend(iter);
        }

        let (lower_bound, _) = iter.size_hint();
        assert!(lower_bound <= isize::MAX as usize); // Ensure offset is indexable
        assert!(index + lower_bound >= index); // Protect against overflow
        assert!(self.len() + lower_bound <= self.capacity());

        let mut num_added = 0;
        let old_len = self.len();
        assert!(index <= old_len);

        unsafe {
            let start = self.as_mut_ptr();
            let ptr = start.add(index);

            // Move the trailing elements.
            ptr::copy(ptr, ptr.add(lower_bound), old_len - index);

            // In case the iterator panics, don't double-drop the items we just copied above.
            self.length = 0;
            let mut guard = DropOnPanic {
                start,
                skip: index..(index + lower_bound),
                len: old_len + lower_bound,
            };

            while num_added < lower_bound {
                let element = match iter.next() {
                    Some(x) => x,
                    None => break,
                };
                let cur = ptr.add(num_added);
                ptr::write(cur, element);
                guard.skip.start += 1;
                num_added += 1;
            }

            if num_added < lower_bound {
                // Iterator provided fewer elements than the hint. Move the tail backward.
                ptr::copy(ptr.add(lower_bound), ptr.add(num_added), old_len - index);
            }
            // There are no more duplicate or uninitialized slots, so the guard is not needed.
            self.set_len(old_len + num_added);
            mem::forget(guard);

            // Insert any remaining elements one-by-one.
            for element in iter {
                self.insert(index + num_added, element);
                num_added += 1;
            }
        }
    }

    /// Convert a StackVec to a Vec.
    pub fn into_vec(self) -> Vec<T> {
        self.into_iter().collect()
    }

    /// Convert the StackVec into a `[T; N]`.
    pub fn into_inner(self) -> Result<[T; N], Self> {
        if self.len() != N {
            Err(self)
        } else {
            unsafe {
                let this = mem::ManuallyDrop::new(self);
                let array = ptr::read(this.as_ptr() as *const [T; N]);
                Ok(array)
            }
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` such that `f(&e)` returns `false`.
    /// This method operates in place and preserves the order of the retained
    /// elements.
    pub fn retain<F: FnMut(&mut T) -> bool>(&mut self, mut f: F) {
        let mut del = 0;
        let len = self.len();
        for i in 0..len {
            if !f(&mut self[i]) {
                del += 1;
            } else if del > 0 {
                self.swap(i - del, i);
            }
        }
        self.truncate(len - del);
    }

    /// Removes consecutive duplicate elements.
    pub fn dedup(&mut self)
    where
        T: PartialEq<T>,
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Removes consecutive duplicate elements using the given equality relation.
    pub fn dedup_by<F>(&mut self, mut same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        // See the implementation of Vec::dedup_by in the
        // standard library for an explanation of this algorithm.
        let len = self.len();
        assert!(len <= self.data.len());
        if len <= 1 {
            return;
        }

        // NOTE: Not ideal but use pointers since the type checker cannot guarantee
        // that the derefs point to different locations in the array.
        let ptr = self.as_mut_ptr();
        let mut w: usize = 1;

        unsafe {
            for r in 1..len {
                let p_r = ptr.add(r);
                let p_wm1 = ptr.add(w - 1);
                if !same_bucket(&mut *p_r, &mut *p_wm1) {
                    if r != w {
                        let p_w = p_wm1.offset(1);
                        ptr::swap(p_r, p_w);
                    }
                    w += 1;
                }
            }
        }

        self.truncate(w);
    }

    /// Removes consecutive elements that map to the same key.
    pub fn dedup_by_key<F, K>(&mut self, mut key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq<K>,
    {
        self.dedup_by(|a, b| key(a) == key(b));
    }
}

impl<T, const N: usize> StackVec<T, N>
where
    T: Copy,
{
    /// Copy the elements from a slice into a new `StackVec`.
    ///
    /// For slices of `Copy` types, this is more efficient than `StackVec::from(slice)`.
    pub fn from_slice(slice: &[T]) -> Self {
        assert!(slice.len() <= N);
        let mut v = StackVec::new();
        unsafe {
            let mut local_len = SetLenOnDrop::new(&mut v.length);
            for (index, item) in slice.iter().enumerate() {
                v.data[index].write(*item);
                local_len.increment_len(1);
            }
        }
        v
    }

    /// Copy elements from a slice into the vector at position `index`, shifting any following
    /// elements toward the back.
    ///
    /// For slices of `Copy` types, this is more efficient than `insert`.
    pub fn insert_from_slice(&mut self, index: usize, slice: &[T]) {
        // NOTE: Cannot overflow, since the number of bytes of both must be <= isize::MAX,
        // so we cannot have unsigned integer wrapping.
        //  https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety
        assert!(index <= self.len() && self.len() + slice.len() <= self.capacity());
        let len = self.len();
        // NOTE: this is safe since the length isn't set, so there won't be duplicate drops
        let ptr = unsafe { self.as_mut_ptr().add(index) };
        unsafe { ptr::copy(ptr, ptr.add(slice.len()), len - index) };

        let mut local_len = SetLenOnDrop::new(&mut self.length);
        for (i, item) in slice.iter().enumerate() {
            self.data[index + i].write(*item);
            unsafe { local_len.increment_len(1) };
        }
    }

    /// Copy elements from a slice and append them to the vector.
    ///
    /// For slices of `Copy` types, this is more efficient than `extend`.
    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        // NOTE: Cannot overflow, since the number of bytes of both must be <= isize::MAX,
        // so we cannot have unsigned integer wrapping.
        //  https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety
        assert!(self.len() + slice.len() <= self.capacity());
        let len = self.len();
        let mut local_len = SetLenOnDrop::new(&mut self.length);
        for (i, item) in slice.iter().enumerate() {
            self.data[len + i].write(*item);
            unsafe { local_len.increment_len(1) };
        }
    }
}

impl<T, const N: usize> StackVec<T, N>
where
    T: Clone,
{
    /// Resizes the vector so that its length is equal to `len`.
    ///
    /// If `len` is less than the current length, the vector simply truncated.
    ///
    /// If `len` is greater than the current length, `value` is appended to the
    /// vector until its length equals `len`.
    pub fn resize(&mut self, len: usize, value: T) {
        assert!(len <= self.capacity());
        let old_len = self.len();
        if len > old_len {
            self.extend(iter::repeat(value).take(len - old_len));
        } else {
            self.truncate(len);
        }
    }

    /// Creates a `StackVec` with `n` copies of `elem`.
    /// ```
    /// use stackvector::StackVec;
    ///
    /// let v = StackVec::<char, 128>::from_elem('d', 2);
    /// assert_eq!(v, StackVec::from_buf(['d', 'd']));
    /// ```
    pub fn from_elem(elem: T, n: usize) -> Self {
        assert!(n <= N);
        let mut v = StackVec::<T, N>::new();
        {
            let mut local_len = SetLenOnDrop::new(&mut v.length);
            for i in 0..n {
                v.data[i].write(elem.clone());
                unsafe { local_len.increment_len(1) };
            }
        }
        v
    }
}

impl<T, const N: usize> ops::Deref for StackVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        unsafe {
            let ptr = self.as_ptr();
            slice::from_raw_parts(ptr, self.len())
        }
    }
}

impl<T, const N: usize> ops::DerefMut for StackVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            let ptr = self.as_mut_ptr();
            slice::from_raw_parts_mut(ptr, self.len())
        }
    }
}

impl<T, const N: usize> AsRef<[T]> for StackVec<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> AsMut<[T]> for StackVec<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, const N: usize> Borrow<[T]> for StackVec<T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> BorrowMut<[T]> for StackVec<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

#[cfg(feature = "std")]
impl<const N: usize> ::std::io::Write for StackVec<u8, N> {
    #[inline]
    fn write(&mut self, buf: &[u8]) -> ::std::io::Result<usize> {
        self.extend_from_slice(buf);
        Ok(buf.len())
    }

    #[inline]
    fn write_all(&mut self, buf: &[u8]) -> ::std::io::Result<()> {
        self.extend_from_slice(buf);
        Ok(())
    }

    #[inline]
    fn flush(&mut self) -> ::std::io::Result<()> {
        Ok(())
    }
}

impl<'a, T, const N: usize> From<&'a [T]> for StackVec<T, N>
where
    T: Clone,
{
    #[inline]
    fn from(slice: &'a [T]) -> StackVec<T, N> {
        slice.iter().cloned().collect()
    }
}

impl<T, const N: usize> From<Vec<T>> for StackVec<T, N> {
    #[inline]
    fn from(vec: Vec<T>) -> StackVec<T, N> {
        StackVec::from_vec(vec)
    }
}

impl<T, const N: usize> From<[T; N]> for StackVec<T, N> {
    #[inline]
    fn from(array: [T; N]) -> StackVec<T, N> {
        StackVec::from_buf(array)
    }
}

macro_rules! impl_index {
    ($index_type: ty, $output_type: ty) => {
        impl<T, const N: usize> ops::Index<$index_type> for StackVec<T, N> {
            type Output = $output_type;
            #[inline]
            fn index(&self, index: $index_type) -> &$output_type {
                &self.as_slice()[index]
            }
        }

        impl<T, const N: usize> ops::IndexMut<$index_type> for StackVec<T, N> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut $output_type {
                &mut self.as_mut_slice()[index]
            }
        }
    };
}

impl_index!(usize, T);
impl_index!(ops::Range<usize>, [T]);
impl_index!(ops::RangeFrom<usize>, [T]);
impl_index!(ops::RangeFull, [T]);
impl_index!(ops::RangeTo<usize>, [T]);
impl_index!(ops::RangeInclusive<usize>, [T]);
impl_index!(ops::RangeToInclusive<usize>, [T]);

impl<T, const N: usize> ExtendFromSlice<T> for StackVec<T, N>
where
    T: Copy,
{
    fn extend_from_slice(&mut self, other: &[T]) {
        StackVec::extend_from_slice(self, other)
    }
}

impl<T, const N: usize> VecLike<T> for StackVec<T, N> {
    #[inline]
    fn push(&mut self, value: T) {
        StackVec::push(self, value);
    }

    #[inline]
    fn pop(&mut self) -> Option<T> {
        StackVec::pop(self)
    }
}

impl<T, const N: usize> iter::FromIterator<T> for StackVec<T, N> {
    fn from_iter<I: iter::IntoIterator<Item = T>>(iterable: I) -> StackVec<T, N> {
        let mut v = StackVec::new();
        v.extend(iterable);
        v
    }
}

impl<T, const N: usize> Extend<T> for StackVec<T, N> {
    fn extend<I: iter::IntoIterator<Item = T>>(&mut self, iterable: I) {
        // size_hint() has no safety guarantees, and TrustedLen
        // is nightly only, so we can't do any optimizations with
        // size_hint.
        for elem in iterable.into_iter() {
            self.push(elem);
        }
    }
}

impl<T, const N: usize> fmt::Debug for StackVec<T, N>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, const N: usize> Default for StackVec<T, N> {
    #[inline]
    fn default() -> StackVec<T, N> {
        StackVec::new()
    }
}

impl<T, const N: usize> Drop for StackVec<T, N> {
    fn drop(&mut self) {
        unsafe {
            // NOTE: as a precaution to avoid duplicate drops
            let len = self.len();
            let mut local_len = SetLenOnDrop::new(&mut self.length);
            for item in &mut self.data[..len] {
                item.assume_init_drop();
                local_len.decrement_len(1);
            }
        }
    }
}

impl<T, const N: usize> Clone for StackVec<T, N>
where
    T: Clone,
{
    fn clone(&self) -> StackVec<T, N> {
        let mut v = StackVec::new();
        for element in self.iter() {
            v.push(element.clone())
        }
        v
    }
}

impl<T, U, const TN: usize, const UN: usize> PartialEq<StackVec<U, UN>> for StackVec<T, TN>
where
    T: PartialEq<U>,
{
    #[inline]
    fn eq(&self, other: &StackVec<U, UN>) -> bool {
        self[..] == other[..]
    }
}

impl<T, const N: usize> Eq for StackVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialOrd for StackVec<T, N>
where
    T: PartialOrd,
{
    #[inline]
    fn partial_cmp(&self, other: &StackVec<T, N>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<T, const N: usize> Ord for StackVec<T, N>
where
    T: Ord,
{
    #[inline]
    fn cmp(&self, other: &StackVec<T, N>) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<T, const N: usize> hash::Hash for StackVec<T, N>
where
    T: hash::Hash,
{
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

unsafe impl<T, const N: usize> Send for StackVec<T, N> where T: Send {}

/// An iterator that consumes a `StackVec` and yields its items by value.
///
/// Returned from [`StackVec::into_iter`][1].
///
/// [1]: struct.StackVec.html#method.into_iter
pub struct IntoIter<T, const N: usize> {
    data: StackVec<T, N>,
    current: usize,
    end: usize,
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                let current = self.current;
                self.current += 1;
                Some(ptr::read(self.data.as_ptr().add(current)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.current;
        (size, Some(size))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                self.end -= 1;
                Some(ptr::read(self.data.as_ptr().add(self.end)))
            }
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}

impl<T, const N: usize> IntoIterator for StackVec<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = T;
    fn into_iter(mut self) -> Self::IntoIter {
        unsafe {
            // Set StackVec len to zero as `IntoIter` drop handles dropping of the elements
            let len = self.len();
            self.set_len(0);
            IntoIter {
                data: self,
                current: 0,
                end: len,
            }
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a StackVec<T, N> {
    type IntoIter = slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut StackVec<T, N> {
    type IntoIter = slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// STACKVEC MACRO

/// Creates a [`StackVec`] containing the arguments.
///
/// `stackvec!` allows `StackVec`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`StackVec`] containing a given list of elements:
///
/// ```
/// # #[macro_use] extern crate stackvector;
/// # use stackvector::StackVec;
/// # fn main() {
/// let v: StackVec<i32, 128> = stackvec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # }
/// ```
///
/// - Create a [`StackVec`] from a given element and size:
///
/// ```
/// # #[macro_use] extern crate stackvector;
/// # use stackvector::StackVec;
/// # fn main() {
/// let v: StackVec<i32, 0x8000> = stackvec![1; 3];
/// assert_eq!(v, StackVec::from_buf([1, 1, 1]));
/// # }
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `stackvec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to independently
/// boxed integers.
#[macro_export]
macro_rules! stackvec {
    // count helper: transform any expression into 1
    (@one $x:expr) => (1usize);
    ($elem:expr; $n:expr) => ({
        $crate::StackVec::from_elem($elem, $n)
    });
    ($($x:expr),*$(,)*) => ({
        // Allow an unused mut variable, since if the sequence is empty,
        // the vec will never be mutated.
        #[allow(unused_mut)] {
            let mut vec = $crate::StackVec::new();
            $(vec.push($x);)*
            vec
        }
    });
}

// TESTS
// -----

#[cfg(test)]
mod test {
    use super::*;
    use alloc::borrow::ToOwned;
    use alloc::boxed::Box;
    use alloc::rc::Rc;
    use alloc::string::String;
    use alloc::vec;
    use core::iter::FromIterator;

    struct BadBoundsIterator1(u8);

    impl BadBoundsIterator1 {
        pub fn new() -> Self {
            BadBoundsIterator1(0)
        }
    }

    impl Iterator for BadBoundsIterator1 {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            self.0 += 1;
            if self.0 >= 10 {
                None
            } else {
                Some(0x41)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let lower_bound = 20;
            let upper_bound = Some(0);
            (lower_bound, upper_bound)
        }
    }

    struct BadBoundsIterator2(u8);

    impl BadBoundsIterator2 {
        pub fn new() -> Self {
            BadBoundsIterator2(0)
        }
    }

    impl Iterator for BadBoundsIterator2 {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            self.0 += 1;
            if self.0 >= 30 {
                None
            } else {
                Some(0x41)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let lower_bound = 0;
            let upper_bound = Some(0);
            (lower_bound, upper_bound)
        }
    }

    struct BadSizeHint(u8);

    impl BadSizeHint {
        pub fn new(start: u8) -> Self {
            BadSizeHint(start)
        }
    }

    impl Iterator for BadSizeHint {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            self.0 += 1;
            if self.0 >= 30 {
                None
            } else {
                Some(0x41)
            }
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let lower_bound = 0;
            let upper_bound = None;
            (lower_bound, upper_bound)
        }
    }

    #[test]
    pub fn test_zero() {
        let v = StackVec::<usize, 0>::new();
        assert_eq!(v.len(), 0);
    }

    #[test]
    #[should_panic]
    pub fn test_panic() {
        let mut v = StackVec::<usize, 0>::new();
        v.push(0);
    }

    // We heap allocate all these strings so that double frees will show up under valgrind.

    #[test]
    pub fn test_inline() {
        let mut v = StackVec::<String, 16>::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        assert_eq!(&*v, &["hello".to_owned(), "there".to_owned(),][..]);
    }

    #[test]
    #[should_panic]
    pub fn test_spill() {
        let mut v = StackVec::<String, 2>::new();
        v.push("hello".to_owned());
        assert_eq!(v[0], "hello");
        v.push("there".to_owned());
        v.push("burma".to_owned());
        assert_eq!(v[0], "hello");
        v.push("shave".to_owned());
        assert_eq!(
            &*v,
            &[
                "hello".to_owned(),
                "there".to_owned(),
                "burma".to_owned(),
                "shave".to_owned(),
            ][..]
        );
    }

    #[test]
    #[should_panic]
    pub fn test_double_spill() {
        let mut v = StackVec::<String, 2>::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        assert_eq!(
            &*v,
            &[
                "hello".to_owned(),
                "there".to_owned(),
                "burma".to_owned(),
                "shave".to_owned(),
                "hello".to_owned(),
                "there".to_owned(),
                "burma".to_owned(),
                "shave".to_owned(),
            ][..]
        );
    }

    /// https://github.com/servo/rust-smallvec/issues/4
    #[test]
    fn issue_4() {
        StackVec::<Box<u32>, 2>::new();
    }

    /// https://github.com/servo/rust-smallvec/issues/5
    #[test]
    fn issue_5() {
        assert!(Some(StackVec::<&u32, 2>::new()).is_some());
    }

    #[test]
    fn drain_test() {
        let mut v: StackVec<u8, 2> = StackVec::new();
        v.push(3);
        assert_eq!(v.drain().collect::<Vec<_>>(), &[3]);
    }

    #[test]
    fn drain_rev_test() {
        let mut v: StackVec<u8, 2> = StackVec::new();
        v.push(3);
        assert_eq!(v.drain().rev().collect::<Vec<_>>(), &[3]);
    }

    #[test]
    fn into_iter() {
        let mut v: StackVec<u8, 2> = StackVec::new();
        v.push(3);
        assert_eq!(v.into_iter().collect::<Vec<_>>(), &[3]);
    }

    #[test]
    fn into_iter_rev() {
        let mut v: StackVec<u8, 2> = StackVec::new();
        v.push(3);
        assert_eq!(v.into_iter().rev().collect::<Vec<_>>(), &[3]);
    }

    #[test]
    fn into_iter_drop() {
        use core::cell::Cell;

        struct DropCounter<'a>(&'a Cell<i32>);

        impl<'a> Drop for DropCounter<'a> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        {
            let cell = Cell::new(0);
            let mut v: StackVec<DropCounter, 2> = StackVec::new();
            v.push(DropCounter(&cell));
            v.into_iter();
            assert_eq!(cell.get(), 1);
        }

        {
            let cell = Cell::new(0);
            let mut v: StackVec<DropCounter, 2> = StackVec::new();
            v.push(DropCounter(&cell));
            v.push(DropCounter(&cell));
            assert!(v.into_iter().next().is_some());
            assert_eq!(cell.get(), 2);
        }
    }

    #[test]
    fn test_capacity() {
        let v: StackVec<u8, 2> = StackVec::new();
        assert_eq!(v.capacity(), 2);
    }

    #[test]
    fn test_truncate() {
        let mut v: StackVec<Box<u8>, 8> = StackVec::new();

        for x in 0..8 {
            v.push(Box::new(x));
        }
        v.truncate(4);

        assert_eq!(v.len(), 4);

        assert_eq!(*v.swap_remove(1), 1);
        assert_eq!(*v.remove(1), 3);
        v.insert(1, Box::new(3));

        assert_eq!(&v.iter().map(|v| **v).collect::<Vec<_>>(), &[0, 3, 2]);
    }

    #[test]
    fn test_insert_many() {
        let mut v: StackVec<u8, 8> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        assert_eq!(v.len(), 4);
        v.insert_many(1, [5, 6].iter().cloned());
        assert_eq!(
            &v.iter().map(|v| *v).collect::<Vec<_>>(),
            &[0, 5, 6, 1, 2, 3]
        );
    }

    #[test]
    fn test_insert_many_buggy_iterator() {
        let mut v: StackVec<u8, 64> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        v.insert_many(1, BadBoundsIterator1::new());
        assert_eq!(
            &v.iter().map(|v| *v).collect::<Vec<_>>(),
            &[0, 65, 65, 65, 65, 65, 65, 65, 65, 65, 1, 2, 3]
        );

        let mut v: StackVec<u8, 64> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        v.insert_many(1, BadBoundsIterator2::new());
        assert_eq!(v.len(), 33);

        let mut v: StackVec<u8, 64> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        v.insert_many(1, BadSizeHint::new(1));
        assert_eq!(v.len(), 32);
    }

    #[should_panic]
    #[test]
    fn test_insert_many_panic_buggy_iterator() {
        let mut v: StackVec<u8, 8> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        v.insert_many(1, BadBoundsIterator2::new());
    }

    #[test]
    fn test_insert_from_slice() {
        let mut v: StackVec<u8, 8> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        assert_eq!(v.len(), 4);
        v.insert_from_slice(1, &[5, 6]);
        assert_eq!(
            &v.iter().map(|v| *v).collect::<Vec<_>>(),
            &[0, 5, 6, 1, 2, 3]
        );
    }

    #[test]
    fn test_extend_from_slice() {
        let mut v: StackVec<u8, 8> = StackVec::new();
        for x in 0..4 {
            v.push(x);
        }
        assert_eq!(v.len(), 4);
        v.extend_from_slice(&[5, 6]);
        assert_eq!(
            &v.iter().map(|v| *v).collect::<Vec<_>>(),
            &[0, 1, 2, 3, 5, 6]
        );
    }

    #[test]
    #[should_panic]
    fn test_drop_panic_smallvec() {
        // This test should only panic once, and not double panic,
        // which would mean a double drop
        struct DropPanic;

        impl Drop for DropPanic {
            fn drop(&mut self) {
                panic!("drop");
            }
        }

        let mut v = StackVec::<DropPanic, 1>::new();
        v.push(DropPanic);
    }

    #[test]
    fn test_eq() {
        let mut a: StackVec<u32, 2> = StackVec::new();
        let mut b: StackVec<u32, 2> = StackVec::new();
        let mut c: StackVec<u32, 2> = StackVec::new();
        // a = [1, 2]
        a.push(1);
        a.push(2);
        // b = [1, 2]
        b.push(1);
        b.push(2);
        // c = [3, 4]
        c.push(3);
        c.push(4);

        assert!(a == b);
        assert!(a != c);
    }

    #[test]
    fn test_ord() {
        let mut a: StackVec<u32, 2> = StackVec::new();
        let mut b: StackVec<u32, 2> = StackVec::new();
        let mut c: StackVec<u32, 2> = StackVec::new();
        // a = [1]
        a.push(1);
        // b = [1, 1]
        b.push(1);
        b.push(1);
        // c = [1, 2]
        c.push(1);
        c.push(2);

        assert!(a < b);
        assert!(b > a);
        assert!(b < c);
        assert!(c > b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;

        {
            let mut a: StackVec<u32, 2> = StackVec::new();
            let b = [1, 2];
            a.extend(b.iter().cloned());
            let mut hasher = DefaultHasher::new();
            assert_eq!(a.hash(&mut hasher), b.hash(&mut hasher));
        }
        {
            let mut a: StackVec<u32, 4> = StackVec::new();
            let b = [1, 2, 11, 12];
            a.extend(b.iter().cloned());
            let mut hasher = DefaultHasher::new();
            assert_eq!(a.hash(&mut hasher), b.hash(&mut hasher));
        }
    }

    #[test]
    fn test_as_ref() {
        let mut a: StackVec<u32, 3> = StackVec::new();
        a.push(1);
        assert_eq!(a.as_ref(), [1]);
        a.push(2);
        assert_eq!(a.as_ref(), [1, 2]);
        a.push(3);
        assert_eq!(a.as_ref(), [1, 2, 3]);
    }

    #[test]
    fn test_as_mut() {
        let mut a: StackVec<u32, 3> = StackVec::new();
        a.push(1);
        assert_eq!(a.as_mut(), [1]);
        a.push(2);
        assert_eq!(a.as_mut(), [1, 2]);
        a.push(3);
        assert_eq!(a.as_mut(), [1, 2, 3]);
        a.as_mut()[1] = 4;
        assert_eq!(a.as_mut(), [1, 4, 3]);
    }

    #[test]
    fn test_borrow() {
        use core::borrow::Borrow;

        let mut a: StackVec<u32, 3> = StackVec::new();
        a.push(1);
        assert_eq!(a.borrow(), [1]);
        a.push(2);
        assert_eq!(a.borrow(), [1, 2]);
        a.push(3);
        assert_eq!(a.borrow(), [1, 2, 3]);
    }

    #[test]
    fn test_borrow_mut() {
        use core::borrow::BorrowMut;

        let mut a: StackVec<u32, 3> = StackVec::new();
        a.push(1);
        assert_eq!(a.borrow_mut(), [1]);
        a.push(2);
        assert_eq!(a.borrow_mut(), [1, 2]);
        a.push(3);
        assert_eq!(a.borrow_mut(), [1, 2, 3]);
        BorrowMut::<[u32]>::borrow_mut(&mut a)[1] = 4;
        assert_eq!(a.borrow_mut(), [1, 4, 3]);
    }

    #[test]
    fn test_from() {
        assert_eq!(&StackVec::<u32, 2>::from(&[1][..])[..], [1]);
        assert_eq!(&StackVec::<u32, 3>::from(&[1, 2, 3][..])[..], [1, 2, 3]);

        let vec = vec![];
        let stack_vec: StackVec<u8, 3> = StackVec::from(vec);
        assert_eq!(&*stack_vec, &[]);
        drop(stack_vec);

        let vec = vec![1, 2, 3, 4, 5];
        let stack_vec: StackVec<u8, 5> = StackVec::from(vec);
        assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
        drop(stack_vec);

        let vec = vec![1, 2, 3, 4, 5];
        let stack_vec: StackVec<u8, 5> = StackVec::from(vec);
        assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
        drop(stack_vec);

        let array = [1];
        let stack_vec: StackVec<u8, 1> = StackVec::from(array);
        assert_eq!(&*stack_vec, &[1]);
        drop(stack_vec);

        let array = [99; 128];
        let stack_vec: StackVec<u8, 128> = StackVec::from(array);
        assert_eq!(&*stack_vec, vec![99u8; 128].as_slice());
        drop(stack_vec);
    }

    #[test]
    fn test_from_slice() {
        assert_eq!(&StackVec::<u32, 2>::from_slice(&[1][..])[..], [1]);
        assert_eq!(
            &StackVec::<u32, 3>::from_slice(&[1, 2, 3][..])[..],
            [1, 2, 3]
        );
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut vec = StackVec::<u32, 3>::from(&[1, 2, 3][..]);
        assert_eq!(vec.clone().into_iter().len(), 3);
        assert_eq!(vec.drain().len(), 3);
    }

    #[test]
    fn veclike_deref_slice() {
        use super::VecLike;

        fn test<T: VecLike<i32>>(vec: &mut T) {
            assert!(!vec.is_empty());
            assert_eq!(vec.len(), 3);

            vec.sort();
            assert_eq!(&vec[..], [1, 2, 3]);
        }

        let mut vec = StackVec::<i32, 3>::from(&[3, 1, 2][..]);
        test(&mut vec);
    }

    #[test]
    fn test_into_vec() {
        let vec = StackVec::<u8, 2>::from_iter(0..2);
        assert_eq!(vec.into_vec(), vec![0, 1]);

        let vec = StackVec::<u8, 3>::from_iter(0..3);
        assert_eq!(vec.into_vec(), vec![0, 1, 2]);
    }

    #[test]
    fn test_into_inner() {
        let vec = StackVec::<u8, 2>::from_iter(0..2);
        assert_eq!(vec.into_inner(), Ok([0, 1]));

        let vec = StackVec::<u8, 2>::from_iter(0..1);
        assert_eq!(vec.clone().into_inner(), Err(vec));

        let vec = StackVec::<u8, 3>::from_iter(0..3);
        assert_eq!(vec.clone().into_inner(), Ok([0, 1, 2]));

        let vec = StackVec::<u8, 4>::from_iter(0..3);
        assert_eq!(vec.clone().into_inner(), Err(vec));
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![];
        let stack_vec: StackVec<u8, 3> = StackVec::from_vec(vec);
        assert_eq!(&*stack_vec, &[]);
        drop(stack_vec);

        let vec = vec![];
        let stack_vec: StackVec<u8, 1> = StackVec::from_vec(vec);
        assert_eq!(&*stack_vec, &[]);
        drop(stack_vec);

        let vec = vec![1];
        let stack_vec: StackVec<u8, 3> = StackVec::from_vec(vec);
        assert_eq!(&*stack_vec, &[1]);
        drop(stack_vec);

        let vec = vec![1, 2, 3];
        let stack_vec: StackVec<u8, 3> = StackVec::from_vec(vec);
        assert_eq!(&*stack_vec, &[1, 2, 3]);
        drop(stack_vec);

        let vec = vec![1, 2, 3, 4, 5];
        let stack_vec: StackVec<u8, 5> = StackVec::from_vec(vec);
        assert_eq!(&*stack_vec, &[1, 2, 3, 4, 5]);
        drop(stack_vec);
    }

    #[test]
    fn test_retain() {
        let mut sv: StackVec<i32, 5> = StackVec::from_slice(&[1, 2, 3, 3, 4]);
        sv.retain(|&mut i| i != 3);
        assert_eq!(sv.pop(), Some(4));
        assert_eq!(sv.pop(), Some(2));
        assert_eq!(sv.pop(), Some(1));
        assert_eq!(sv.pop(), None);

        // Test that drop implementations are called for inline.
        let one = Rc::new(1);
        let mut sv: StackVec<Rc<i32>, 3> = StackVec::new();
        sv.push(Rc::clone(&one));
        assert_eq!(Rc::strong_count(&one), 2);
        sv.retain(|_| false);
        assert_eq!(Rc::strong_count(&one), 1);
    }

    #[test]
    fn test_dedup() {
        let mut dupes: StackVec<i32, 5> = StackVec::from_slice(&[1, 1, 2, 3, 3]);
        dupes.dedup();
        assert_eq!(&*dupes, &[1, 2, 3]);

        let mut empty: StackVec<i32, 5> = StackVec::new();
        empty.dedup();
        assert!(empty.is_empty());

        let mut all_ones: StackVec<i32, 5> = StackVec::from_slice(&[1, 1, 1, 1, 1]);
        all_ones.dedup();
        assert_eq!(all_ones.len(), 1);

        let mut no_dupes: StackVec<i32, 5> = StackVec::from_slice(&[1, 2, 3, 4, 5]);
        no_dupes.dedup();
        assert_eq!(no_dupes.len(), 5);
    }

    #[test]
    fn test_resize() {
        let mut v: StackVec<i32, 8> = StackVec::new();
        v.push(1);
        v.resize(5, 0);
        assert_eq!(v[..], [1, 0, 0, 0, 0][..]);

        v.resize(2, -1);
        assert_eq!(v[..], [1, 0][..]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_write() {
        use std::io::Write;

        let data = [1, 2, 3, 4, 5];

        let mut small_vec: StackVec<u8, 5> = StackVec::new();
        let len = small_vec.write(&data[..]).unwrap();
        assert_eq!(len, 5);
        assert_eq!(small_vec.as_ref(), data.as_ref());

        let mut small_vec: StackVec<u8, 5> = StackVec::new();
        small_vec.write_all(&data[..]).unwrap();
        assert_eq!(small_vec.as_ref(), data.as_ref());
    }
}
