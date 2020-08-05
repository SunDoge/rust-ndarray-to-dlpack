pub mod dlpack;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use std::ffi::CString;
use std::os::raw::{c_int, c_void};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn eye(n: usize) -> PyResult<*mut pyo3::ffi::PyObject> {
    let mut x: ndarray::ArcArray<f32, ndarray::IxDyn> =
        ndarray::Array::eye(n).into_dyn().into_shared();

    println!("eye = \n{}", x);

    let dlm_tensor = to_dlpack(x.clone());
    let name = CString::new("dltensor").unwrap();

    let ptr = unsafe {
        pyo3::ffi::PyCapsule_New(
            dlm_tensor as *mut c_void,
            name.as_ptr(),
            Some(destructor as pyo3::ffi::PyCapsule_Destructor),
        )
    };

    std::mem::forget(name);
    std::mem::forget(dlm_tensor);
    std::mem::forget(x);

    Ok(ptr)
}

fn array_to_dl_tensor(mut arr: ndarray::ArcArray<f32, ndarray::IxDyn>) -> dlpack::DLTensor {
    dlpack::DLTensor {
        data: arr.as_mut_ptr() as *mut c_void,
        ctx: dlpack::DLContext {
            device_type: dlpack::DLDeviceType_kDLCPU,
            device_id: 0,
        },
        ndim: arr.ndim() as c_int,
        dtype: dlpack::DLDataType {
            code: dlpack::DLDataTypeCode_kDLFloat as u8,
            bits: 32,
            lanes: 1,
        },
        shape: arr.shape().as_ptr() as *const i64 as *mut i64,
        strides: arr.strides().as_ptr() as *const i64 as *mut i64,
        byte_offset: 0,
    }
}

// #[repr(C)]
struct ArrayDLMTensor {
    array: ndarray::ArcArray<f32, ndarray::IxDyn>,
    tensor: dlpack::DLManagedTensor,
}

unsafe extern "C" fn deleter(x: *mut dlpack::DLManagedTensor) {
    println!("DLManagedTensor deleter");

    let ctx: *mut ArrayDLMTensor = (*x).manager_ctx as *mut ArrayDLMTensor;
    std::mem::drop(ctx);
}

unsafe extern "C" fn destructor(o: *mut pyo3::ffi::PyObject) {
    println!("PyCapsule destructor");

    let name = CString::new("dltensor").unwrap();

    let ptr = pyo3::ffi::PyCapsule_GetPointer(o, name.as_ptr()) as *mut dlpack::DLManagedTensor;
    println!("Get Pointer");

    dbg!(*ptr);
    (*ptr).deleter.unwrap()(ptr);
}

fn to_dlpack(src: ndarray::ArcArray<f32, ndarray::IxDyn>) -> *mut dlpack::DLManagedTensor {
    let mut array_dlm_tensor = ArrayDLMTensor {
        array: src.clone(),
        tensor: dlpack::DLManagedTensor {
            dl_tensor: array_to_dl_tensor(src.clone()),
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(deleter),
        },
    };

    array_dlm_tensor.tensor.manager_ctx =
        &mut array_dlm_tensor as *mut ArrayDLMTensor as *mut c_void;

    let dlm_tensor = &mut array_dlm_tensor.tensor as *mut dlpack::DLManagedTensor;

    std::mem::forget(array_dlm_tensor);

    dlm_tensor
}

/// A Python module implemented in Rust.
#[pymodule]
fn string_sum(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sum_as_string))?;
    m.add_wrapped(wrap_pyfunction!(eye))?;
    Ok(())
}
