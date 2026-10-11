//! Over-aligned scratch buffers for packed panels.

use std::alloc::{alloc, dealloc, Layout};

/// Cache-line / AVX-512 friendly alignment for packed panels.
pub(crate) const PANEL_ALIGN: usize = 64;

/// A 64-byte-aligned, uninitialised-on-creation buffer of `T`.
///
/// Packed panels are fully written before being read (the packing routine
/// zero-fills edge rows), so the contents start out as whatever the allocator
/// handed back. `as_mut_ptr` is the only way in.
pub(crate) struct Panel<T> {
    ptr: *mut T,
    layout: Layout,
}

impl<T> Panel<T> {
    pub fn new(len: usize) -> Self {
        let len = len.max(1);
        let layout = Layout::from_size_align(len * core::mem::size_of::<T>(), PANEL_ALIGN)
            .expect("panel layout")
            .pad_to_align();
        // SAFETY: size is non-zero because len >= 1 and T is a float type.
        let ptr = unsafe { alloc(layout) } as *mut T;
        assert!(
            !ptr.is_null(),
            "failed to allocate {} byte panel",
            layout.size()
        );
        Panel { ptr, layout }
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T> Drop for Panel<T> {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `new`.
        unsafe { dealloc(self.ptr as *mut u8, self.layout) }
    }
}

// SAFETY: `Panel` owns its allocation exclusively and `T` is a plain scalar.
unsafe impl<T: Send> Send for Panel<T> {}
