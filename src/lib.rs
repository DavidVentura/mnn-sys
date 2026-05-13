//! Safe Rust bindings for a thin C wrapper around Alibaba MNN.
//!
//! Forked from the `mnn` module of `rust-paddle-ocr` (Apache-2.0).
//! The build system (build.rs, cpp wrapper) and inference engine API are
//! preserved; OCR-specific layers are not part of this crate.

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::ffi::{CStr, CString};
use std::ptr::NonNull;

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/mnn_bindings.rs"));
}

const MNNR_MAX_DIMS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnnError {
    InvalidParameter(String),
    OutOfMemory,
    RuntimeError(String),
    Unsupported,
    ModelLoadFailed(String),
    NullPointer,
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnnError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            MnnError::OutOfMemory => write!(f, "Out of memory"),
            MnnError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            MnnError::Unsupported => write!(f, "Unsupported operation"),
            MnnError::ModelLoadFailed(msg) => write!(f, "Model loading failed: {}", msg),
            MnnError::NullPointer => write!(f, "Null pointer"),
            MnnError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for MnnError {}

pub type Result<T> = std::result::Result<T, MnnError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PrecisionMode {
    #[default]
    Normal = 0,
    Low = 1,
    High = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum DataFormat {
    #[default]
    NCHW = 0,
    NHWC = 1,
    Auto = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum MemoryMode {
    #[default]
    Normal = 0,
    Low = 1,
    High = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    #[default]
    CPU,
    Metal,
    OpenCL,
    OpenGL,
    Vulkan,
    CUDA,
    CoreML,
}

#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub thread_count: i32,
    pub precision_mode: PrecisionMode,
    pub use_cache: bool,
    pub data_format: DataFormat,
    pub backend: Backend,
    pub memory_mode: MemoryMode,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            use_cache: false,
            data_format: DataFormat::NCHW,
            backend: Backend::CPU,
            memory_mode: MemoryMode::Normal,
        }
    }
}

impl InferenceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }

    pub fn with_memory(mut self, memory: MemoryMode) -> Self {
        self.memory_mode = memory;
        self
    }

    fn to_ffi(&self) -> ffi::MNNR_Config {
        ffi::MNNR_Config {
            thread_count: self.thread_count,
            precision_mode: self.precision_mode as i32,
            backend: match self.backend {
                Backend::CPU => 0,
                Backend::Metal => 1,
                Backend::OpenCL => 2,
                Backend::OpenGL => 3,
                Backend::Vulkan => 4,
                Backend::CUDA => 5,
                Backend::CoreML => 6,
            },
            use_cache: self.use_cache,
            data_format: self.data_format as i32,
            memory_mode: self.memory_mode as i32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorType {
    F32 = 0,
    I32 = 1,
    I64 = 2,
}

#[derive(Debug, Clone, Copy)]
pub enum TensorData<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl TensorData<'_> {
    fn data_type(self) -> TensorType {
        match self {
            TensorData::F32(_) => TensorType::F32,
            TensorData::I32(_) => TensorType::I32,
            TensorData::I64(_) => TensorType::I64,
        }
    }

    fn len(self) -> usize {
        match self {
            TensorData::F32(data) => data.len(),
            TensorData::I32(data) => data.len(),
            TensorData::I64(data) => data.len(),
        }
    }

    fn as_ptr(self) -> *const std::ffi::c_void {
        match self {
            TensorData::F32(data) => data.as_ptr() as *const std::ffi::c_void,
            TensorData::I32(data) => data.as_ptr() as *const std::ffi::c_void,
            TensorData::I64(data) => data.as_ptr() as *const std::ffi::c_void,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NamedInput<'a> {
    pub name: &'a str,
    pub data: TensorData<'a>,
    pub shape: &'a [usize],
}

#[derive(Debug, Clone)]
pub struct NamedOutput {
    pub name: String,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct SharedRuntime {
    ptr: NonNull<ffi::MNN_SharedRuntime>,
}

impl SharedRuntime {
    pub fn new(config: &InferenceConfig) -> Result<Self> {
        let c_config = config.to_ffi();
        let runtime_ptr = unsafe { ffi::mnnr_create_runtime(&c_config) };
        let ptr = NonNull::new(runtime_ptr)
            .ok_or_else(|| MnnError::RuntimeError("Create shared runtime failed".to_string()))?;
        Ok(SharedRuntime { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::MNN_SharedRuntime {
        self.ptr.as_ptr()
    }
}

impl Drop for SharedRuntime {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_runtime(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for SharedRuntime {}
unsafe impl Sync for SharedRuntime {}

fn get_last_error_message(engine: Option<*const ffi::MNN_InferenceEngine>) -> String {
    match engine {
        Some(ptr) => unsafe {
            let c_str = ffi::mnnr_get_last_error(ptr);
            if c_str.is_null() {
                "Unknown error".to_string()
            } else {
                CStr::from_ptr(c_str).to_string_lossy().into_owned()
            }
        },
        None => "Engine creation failed".to_string(),
    }
}

fn get_last_module_error_message(module: Option<*const ffi::MNN_ModuleEngine>) -> String {
    match module {
        Some(ptr) => unsafe {
            let c_str = ffi::mnnr_module_get_last_error(ptr);
            if c_str.is_null() {
                "Unknown module error".to_string()
            } else {
                CStr::from_ptr(c_str).to_string_lossy().into_owned()
            }
        },
        None => "Module creation failed".to_string(),
    }
}

pub struct ModuleEngine {
    ptr: NonNull<ffi::MNN_ModuleEngine>,
}

impl ModuleEngine {
    pub fn from_file(
        model_path: impl AsRef<std::path::Path>,
        input_names: &[&str],
        output_names: &[&str],
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        if input_names.is_empty() || output_names.is_empty() {
            return Err(MnnError::InvalidParameter(
                "At least one input and output name is required".to_string(),
            ));
        }

        let path = model_path.as_ref();
        let path = path.to_str().ok_or_else(|| {
            MnnError::ModelLoadFailed("Model path is not valid UTF-8".to_string())
        })?;
        let path = CString::new(path).map_err(|_| {
            MnnError::ModelLoadFailed("Model path contains an interior NUL byte".to_string())
        })?;

        let input_storage: Vec<CString> = input_names
            .iter()
            .map(|name| {
                CString::new(*name).map_err(|_| {
                    MnnError::InvalidParameter(format!(
                        "Input name contains an interior NUL byte: {}",
                        name
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let output_storage: Vec<CString> = output_names
            .iter()
            .map(|name| {
                CString::new(*name).map_err(|_| {
                    MnnError::InvalidParameter(format!(
                        "Output name contains an interior NUL byte: {}",
                        name
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let input_ptrs: Vec<*const std::ffi::c_char> =
            input_storage.iter().map(|s| s.as_ptr()).collect();
        let output_ptrs: Vec<*const std::ffi::c_char> =
            output_storage.iter().map(|s| s.as_ptr()).collect();

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();
        let module_ptr = unsafe {
            ffi::mnnr_create_module_from_file(
                path.as_ptr(),
                input_ptrs.as_ptr(),
                input_ptrs.len(),
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                &c_config,
            )
        };

        let ptr = NonNull::new(module_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_module_error_message(None)))?;
        Ok(Self { ptr })
    }

    pub fn run_named_dynamic(
        &self,
        inputs: &[NamedInput<'_>],
        output_names: &[&str],
    ) -> Result<Vec<NamedOutput>> {
        if inputs.is_empty() {
            return Err(MnnError::InvalidParameter(
                "At least one named input is required".to_string(),
            ));
        }
        if output_names.is_empty() {
            return Err(MnnError::InvalidParameter(
                "At least one named output is required".to_string(),
            ));
        }

        for input in inputs {
            if input.shape.is_empty() || input.shape.len() > MNNR_MAX_DIMS {
                return Err(MnnError::InvalidParameter(format!(
                    "Invalid shape for input {}",
                    input.name
                )));
            }
            let expected: usize = input.shape.iter().product();
            if expected != input.data.len() {
                return Err(MnnError::ShapeMismatch {
                    expected: vec![expected],
                    got: vec![input.data.len()],
                });
            }
        }

        let input_names: Vec<CString> = inputs
            .iter()
            .map(|input| {
                CString::new(input.name).map_err(|_| {
                    MnnError::InvalidParameter(format!(
                        "Input name contains an interior NUL byte: {}",
                        input.name
                    ))
                })
            })
            .collect::<Result<_>>()?;

        let output_name_storage: Vec<CString> = output_names
            .iter()
            .map(|name| {
                CString::new(*name).map_err(|_| {
                    MnnError::InvalidParameter(format!(
                        "Output name contains an interior NUL byte: {}",
                        name
                    ))
                })
            })
            .collect::<Result<_>>()?;

        let c_inputs: Vec<ffi::MNNR_NamedInput> = inputs
            .iter()
            .zip(input_names.iter())
            .map(|(input, name)| ffi::MNNR_NamedInput {
                name: name.as_ptr(),
                data: input.data.as_ptr(),
                element_count: input.data.len(),
                dims: input.shape.as_ptr(),
                ndims: input.shape.len(),
                data_type: input.data.data_type() as i32,
            })
            .collect();

        let mut c_outputs: Vec<ffi::MNNR_NamedOutput> = output_name_storage
            .iter()
            .map(|name| ffi::MNNR_NamedOutput {
                name: name.as_ptr(),
                data: std::ptr::null_mut(),
                element_count: 0,
                dims: [0; MNNR_MAX_DIMS],
                ndims: 0,
            })
            .collect();

        let error_code = unsafe {
            ffi::mnnr_run_module_named_dynamic(
                self.ptr.as_ptr(),
                c_inputs.as_ptr(),
                c_inputs.len(),
                c_outputs.as_mut_ptr(),
                c_outputs.len(),
            )
        };

        if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
            for output in &mut c_outputs {
                if !output.data.is_null() {
                    unsafe {
                        ffi::mnnr_free_output(output.data);
                    }
                    output.data = std::ptr::null_mut();
                }
            }
            return match error_code {
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => {
                    Err(MnnError::InvalidParameter(get_last_module_error_message(
                        Some(self.ptr.as_ptr()),
                    )))
                }
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
                _ => Err(MnnError::RuntimeError(get_last_module_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            };
        }

        let mut outputs = Vec::with_capacity(c_outputs.len());
        for (output_name, c_output) in output_names.iter().zip(c_outputs.into_iter()) {
            if c_output.data.is_null() && c_output.element_count > 0 {
                return Err(MnnError::RuntimeError(format!(
                    "Named output {} returned a null buffer",
                    output_name
                )));
            }

            let data = if c_output.element_count == 0 {
                if !c_output.data.is_null() {
                    unsafe {
                        ffi::mnnr_free_output(c_output.data);
                    }
                }
                Vec::new()
            } else {
                unsafe {
                    let slice = std::slice::from_raw_parts(c_output.data, c_output.element_count);
                    let data = slice.to_vec();
                    ffi::mnnr_free_output(c_output.data);
                    data
                }
            };

            outputs.push(NamedOutput {
                name: (*output_name).to_string(),
                data,
                shape: c_output.dims[..c_output.ndims].to_vec(),
            });
        }

        Ok(outputs)
    }
}

impl Drop for ModuleEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_module(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for ModuleEngine {}
unsafe impl Sync for ModuleEngine {}

pub struct InferenceEngine {
    ptr: NonNull<ffi::MNN_InferenceEngine>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl InferenceEngine {
    pub fn from_buffer(model_buffer: &[u8], config: Option<InferenceConfig>) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter(
                "Model data is empty".to_string(),
            ));
        }

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();

        let engine_ptr = unsafe {
            ffi::mnnr_create_engine(
                model_buffer.as_ptr() as *const _,
                model_buffer.len(),
                &c_config,
            )
        };

        let ptr = NonNull::new(engine_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

        let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

        Ok(InferenceEngine {
            ptr,
            input_shape,
            output_shape,
        })
    }

    pub fn from_file(
        model_path: impl AsRef<std::path::Path>,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        let path = path.to_str().ok_or_else(|| {
            MnnError::ModelLoadFailed("Model path is not valid UTF-8".to_string())
        })?;
        let path = CString::new(path).map_err(|_| {
            MnnError::ModelLoadFailed("Model path contains an interior NUL byte".to_string())
        })?;

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();

        let engine_ptr = unsafe { ffi::mnnr_create_engine_from_file(path.as_ptr(), &c_config) };

        let ptr = NonNull::new(engine_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

        let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

        Ok(InferenceEngine {
            ptr,
            input_shape,
            output_shape,
        })
    }

    pub fn from_buffer_with_runtime(model_buffer: &[u8], runtime: &SharedRuntime) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter(
                "Model data is empty".to_string(),
            ));
        }

        let engine_ptr = unsafe {
            ffi::mnnr_create_engine_with_runtime(
                model_buffer.as_ptr() as *const _,
                model_buffer.len(),
                runtime.as_ptr(),
            )
        };

        let ptr = NonNull::new(engine_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

        let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

        Ok(InferenceEngine {
            ptr,
            input_shape,
            output_shape,
        })
    }

    unsafe fn get_shapes(ptr: *mut ffi::MNN_InferenceEngine) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut input_shape_vec = vec![0usize; 8];
        let mut input_ndims = 0;
        let mut output_shape_vec = vec![0usize; 8];
        let mut output_ndims = 0;

        if ffi::mnnr_get_input_shape(ptr, input_shape_vec.as_mut_ptr(), &mut input_ndims)
            != ffi::MNNR_ErrorCode_MNNR_SUCCESS
        {
            return Err(MnnError::RuntimeError(
                "Failed to get input shape".to_string(),
            ));
        }
        input_shape_vec.truncate(input_ndims);

        if ffi::mnnr_get_output_shape(ptr, output_shape_vec.as_mut_ptr(), &mut output_ndims)
            != ffi::MNNR_ErrorCode_MNNR_SUCCESS
        {
            return Err(MnnError::RuntimeError(
                "Failed to get output shape".to_string(),
            ));
        }
        output_shape_vec.truncate(output_ndims);

        Ok((input_shape_vec, output_shape_vec))
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }

        let input_slice = input_data.as_slice().ok_or_else(|| {
            MnnError::InvalidParameter("Input data must be contiguous".to_string())
        })?;

        let output_size: usize = self.output_shape.iter().product();
        let mut output_buffer = vec![0.0f32; output_size];

        let error_code = unsafe {
            ffi::mnnr_run_inference(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_slice.len(),
                output_buffer.as_mut_ptr(),
                output_buffer.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer).map_err(|e| {
                    MnnError::RuntimeError(format!("Failed to create output array: {}", e))
                })
            }
            ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(MnnError::InvalidParameter(
                get_last_error_message(Some(self.ptr.as_ptr())),
            )),
            ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
            ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
            _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                self.ptr.as_ptr(),
            )))),
        }
    }

    pub fn run_raw(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let expected_input: usize = self.input_shape.iter().product();
        let expected_output: usize = self.output_shape.iter().product();

        if input.len() != expected_input {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_input],
                got: vec![input.len()],
            });
        }

        if output.len() != expected_output {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_output],
                got: vec![output.len()],
            });
        }

        let error_code = unsafe {
            ffi::mnnr_run_inference(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => Ok(()),
            ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(MnnError::InvalidParameter(
                get_last_error_message(Some(self.ptr.as_ptr())),
            )),
            ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
            _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                self.ptr.as_ptr(),
            )))),
        }
    }

    pub(crate) fn as_ptr(&self) -> NonNull<ffi::MNN_InferenceEngine> {
        self.ptr
    }

    pub fn has_dynamic_shape(&self) -> bool {
        self.input_shape.iter().any(|&d| d > 100000)
            || self.output_shape.iter().any(|&d| d > 100000)
    }

    pub fn run_dynamic(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let input_shape: Vec<usize> = input_data.shape().to_vec();
        let input_slice = input_data.as_slice().ok_or_else(|| {
            MnnError::InvalidParameter("Input data must be contiguous".to_string())
        })?;

        let mut output_data: *mut f32 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let mut output_dims = [0usize; 8];
        let mut output_ndims: usize = 0;

        let error_code = unsafe {
            ffi::mnnr_run_inference_dynamic(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_shape.as_ptr(),
                input_shape.len(),
                &mut output_data,
                &mut output_size,
                output_dims.as_mut_ptr(),
                &mut output_ndims,
            )
        };

        if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
            return match error_code {
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            };
        }

        let output_shape: Vec<usize> = output_dims[..output_ndims].to_vec();
        let output_buffer = unsafe {
            let slice = std::slice::from_raw_parts(output_data, output_size);
            let buffer = slice.to_vec();
            ffi::mnnr_free_output(output_data);
            buffer
        };

        ArrayD::from_shape_vec(IxDyn(&output_shape), output_buffer)
            .map_err(|e| MnnError::RuntimeError(format!("Failed to create output array: {}", e)))
    }

    pub fn run_dynamic_raw(
        &self,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut output_data: *mut f32 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let mut output_dims = [0usize; 8];
        let mut output_ndims: usize = 0;

        let error_code = unsafe {
            ffi::mnnr_run_inference_dynamic(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input_shape.as_ptr(),
                input_shape.len(),
                &mut output_data,
                &mut output_size,
                output_dims.as_mut_ptr(),
                &mut output_ndims,
            )
        };

        if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
            return match error_code {
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            };
        }

        let output_shape = output_dims[..output_ndims].to_vec();
        let output_buffer = unsafe {
            let slice = std::slice::from_raw_parts(output_data, output_size);
            let buffer = slice.to_vec();
            ffi::mnnr_free_output(output_data);
            buffer
        };

        Ok((output_buffer, output_shape))
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_engine(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

pub struct SessionPool {
    ptr: NonNull<ffi::MNN_SessionPool>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl SessionPool {
    pub fn new(
        engine: &InferenceEngine,
        pool_size: usize,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        if pool_size == 0 {
            return Err(MnnError::InvalidParameter(
                "Pool size cannot be 0".to_string(),
            ));
        }

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();

        let pool_ptr = unsafe {
            ffi::mnnr_create_session_pool(engine.as_ptr().as_ptr(), pool_size, &c_config)
        };

        let ptr = NonNull::new(pool_ptr)
            .ok_or_else(|| MnnError::RuntimeError("Create session pool failed".to_string()))?;

        Ok(SessionPool {
            ptr,
            input_shape: engine.input_shape.clone(),
            output_shape: engine.output_shape.clone(),
        })
    }

    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }

        let input_slice = input_data.as_slice().ok_or_else(|| {
            MnnError::InvalidParameter("Input data must be contiguous".to_string())
        })?;

        let output_size: usize = self.output_shape.iter().product();
        let mut output_buffer = vec![0.0f32; output_size];

        let error_code = unsafe {
            ffi::mnnr_session_pool_run(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_slice.len(),
                output_buffer.as_mut_ptr(),
                output_buffer.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer).map_err(|e| {
                    MnnError::RuntimeError(format!("Failed to create output array: {}", e))
                })
            }
            _ => Err(MnnError::RuntimeError(
                "Session pool inference failed".to_string(),
            )),
        }
    }

    pub fn available(&self) -> usize {
        unsafe { ffi::mnnr_session_pool_available(self.ptr.as_ptr()) }
    }
}

impl Drop for SessionPool {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_session_pool(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for SessionPool {}
unsafe impl Sync for SessionPool {}

pub fn get_version() -> String {
    unsafe {
        let c_str = ffi::mnnr_get_version();
        if c_str.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.precision_mode, PrecisionMode::Normal);
    }

    #[test]
    fn test_config_builder() {
        let config = InferenceConfig::new()
            .with_threads(8)
            .with_precision(PrecisionMode::High)
            .with_backend(Backend::Metal);

        assert_eq!(config.thread_count, 8);
        assert_eq!(config.precision_mode, PrecisionMode::High);
        assert_eq!(config.backend, Backend::Metal);
    }
}
