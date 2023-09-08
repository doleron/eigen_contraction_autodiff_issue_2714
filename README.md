# eigen_contraction_autodiff_issue_2714
https://gitlab.com/libeigen/eigen/-/issues/2714

## Eigen version

3.4.90

commit 4e598ad259bf9561cd5882326e7cafc585d14f47 (HEAD -> master, origin/master, origin/HEAD)

## current workaround

Initializing objects after memory allocation and destroy objects before deallocation:

```c++
template <bool lhs_inner_dim_contiguous, bool rhs_inner_dim_contiguous, bool rhs_inner_dim_reordered, int Alignment, bool use_output_kernel>
  EIGEN_DEVICE_FUNC void evalGemmPartial(Scalar* buffer, Index k_start, Index k_end, int num_threads) const {
    
    // ...

    typedef typename TensorContractionKernel::BlockMemHandle BlockMemHandle;
    const BlockMemHandle packed_mem =
        kernel.allocate(this->m_device, &blockA, &blockB);

    Index block_mid = mc * kc;
    Index block_end = block_mid + nc * kc;

    Index _LhsScalar_size = sizeof(LhsScalar);
    Index _RhsScalar_size = sizeof(RhsScalar);

    // Initializing

    for (Index block_i = 0; block_i < block_mid; ++block_i) {
      new(packed_mem + block_i*_LhsScalar_size)LhsScalar();
    }
    for (Index block_i = block_mid; block_i < block_end; ++block_i) {
      new(packed_mem + block_i*_LhsScalar_size)RhsScalar();
    }

    // ...

    // Deallocatting

    for (Index block_i = 0; block_i < block_mid; ++block_i) {
      LhsScalar *element = static_cast<LhsScalar*>(packed_mem + block_i*_LhsScalar_size);
      element->~LhsScalar();
    }
    for (Index block_i = block_mid; block_i < block_end; ++block_i) {
      RhsScalar *element = static_cast<RhsScalar*>(packed_mem + block_i*_RhsScalar_size);
      element->~RhsScalar();
    }

    kernel.deallocate(this->m_device, packed_mem);
  }
```

Check the modified TensorContration.h file.