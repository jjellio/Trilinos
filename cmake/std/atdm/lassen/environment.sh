#!/bin/bash

# we'll need to adjust LD_LIRBARY_PATH
function ld_remove(){
  export LD_LIBRARY_PATH=`echo -n $LD_LIBRARY_PATH | awk -v RS=: -v ORS=: '$0 != "'$1'"' | sed 's/:$//'`
}

function ld_prepend(){
  ld_remove "$1";
  export LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
}


echo "Using $ATDM_CONFIG_SYSTEM_NAME compiler stack $ATDM_CONFIG_COMPILER to build $ATDM_CONFIG_BUILD_TYPE code with Kokkos node type $ATDM_CONFIG_NODE_TYPE"
echo "$ATDM_CONFIG_BUILD_NAME"


module purge --silent
if [[ $ATDM_CONFIG_BUILD_NAME =~ ^(.*-)mvapich2(-.*)$  ]]; then
  echo "preparing mvapich2 + clang"

  module load cuda/10.1.243
  module load mvapich2/2020.12.11-cuda-10.1.243 xl lapack/3.9.0-xl-2020.11.12

  # grab XL paths (for use with fortran)
  XL_ROOT=$(dirname -- $(dirname -- $(which xlf)))
  XLF=$(which xlf_r)
  MVAPICH2_ORIG_MPIFC=$(which mpifort)

  module load clang/ibm-11.0.1
  MVAPICH2_ORIG_MPICC=$(which mpicc)
  MVAPICH2_ORIG_MPICXX=$(which mpicxx)


  export MVAPICH2_ROOT="/usr/tce/packages/mvapich2/osu/mvapich2-2020.12.11-cuda-10.1.243"
  export SPECTRUM_ROOT="/usr/tce/packages/spectrum-mpi/ibm/spectrum-mpi-rolling-release"

  echo "updating LD_LIRBARY_PATH with mvapich2 and spectrum"

  ld_prepend "${SPECTRUM_ROOT}/lib"
  ld_prepend "$MVAPICH2_ROOT/lib64"
  echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH

  # Set common MPI wrappers
  export MPICC=$MVAPICH2_ORIG_MPICC
  export MPICXX=$MVAPICH2_ORIG_MPICXX
  export MPIF90=$MVAPICH2_ORIG_MPIFC
else
  echo -n "preparing spectrum "
  module load cuda/10.1.243                  &>/dev/null
  module load xl lapack/3.9.0-xl-2020.11.12  &>/dev/null
  # grab XL paths (for use with fortran)
  XL_ROOT=$(dirname -- $(dirname -- $(which xlf)))
  XLF=$(which xlf_r)
  export MPIF90=$(which mpifort)

  if [[ $ATDM_CONFIG_BUILD_NAME =~ ^(.*-)clang(-.*)$  ]]; then
    echo "+ clang"
    module load clang/ibm-11.0.1 &>/dev/null
    export MPICC=$(which mpicc)
    export MPICXX=$(which mpicxx)
  elif [[ $ATDM_CONFIG_BUILD_NAME =~ ^(.*-)xl(-.*)$  ]]; then
    echo "+ xl"
    export MPICC=$(which mpicc)
    export MPICXX=$(which mpicxx)
    # do the nvcc wrapper song and dance
    export NVCC_WRAPPER_DEFAULT_COMPILER=$(which xlC_r)
    export OMPI_CXX=${ATDM_CONFIG_NVCC_WRAPPER}
    if [ ! -x "$OMPI_CXX" ]; then
      echo "No nvcc_wrapper found"
      return
    fi
    export ATDM_CONFIG_CXX_FLAGS+="-ccbin ${NVCC_WRAPPER_DEFAULT_COMPILER} -qxflag=disable__cplusplusOverride"

    # set the gcc compiler XL  will use for backend to one that handles c++14
    export XLC_USR_CONFIG=/opt/ibm/xlC/16.1.1/etc/xlc.cfg.rhel.7.6.gcc.4.8.5.cuda.10.2.2021.3.25.11.42.47
    export XLF_USR_CONFIG=/opt/ibm/xlf/16.1.1/etc/xlf.cfg.rhel.7.6.gcc.4.8.5.cuda.10.2.2021.3.25.11.42.47
  else
    echo "+ gcc"
    module load gcc/7.3.1 &>/dev/null
    export MPIF90=$(which mpif90)
    export MPICC=$(which mpicc)
    export MPICXX=$(which mpicxx)
    
    # do the nvcc wrapper song and dance
    export NVCC_WRAPPER_DEFAULT_COMPILER=$(which g++)
    export OMPI_CXX=${ATDM_CONFIG_NVCC_WRAPPER}
    if [ ! -x "$OMPI_CXX" ]; then
      echo "No nvcc_wrapper found"
      return
    fi
  fi
fi
# Set up stuff related to CUDA
export CUDA_BIN_PATH=$CUDA_HOME

module unload cmake
PATH=$HOME/src/spack/opt/spack/linux-rhel7-power9le/clang-11.0.1/cmake-3.20.1-zirs5jr36wi52rknvpbv3ip6pcc6h4fz/bin:$HOME/src/spack/opt/spack/linux-rhel7-power9le/clang-11.0.1/ninja-kitware-yd63vzfafziohm35cc33bkb6lji5hb3f/bin:$PATH

export ATDM_CONFIG_KOKKOS_ARCH=Power9,Volta70
unset ATDM_CONFIG_CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS


# Some basic settings
export ATDM_CONFIG_ENABLE_SPARC_SETTINGS=OFF
export ATDM_CONFIG_BUILD_COUNT=60
export ATDM_CONFIG_CTEST_PARALLEL_LEVEL=4


# ATDM Settings
export ATDM_CONFIG_USE_CUDA=ON
export ATDM_CONFIG_USE_OPENMP=OFF
export ATDM_CONFIG_USE_PTHREADS=OFF
# Kokkos Settings
export ATDM_CONFIG_Kokkos_ENABLE_SERIAL=ON
export KOKKOS_NUM_DEVICES=4

# Set a standard git so everyone has the same git
module load git/2.20.0

# ATDM specific config variables
export ATDM_CONFIG_LAPACK_LIBS="-L${LAPACK_DIR};-llapack"
export ATDM_CONFIG_BLAS_LIBS="-L${LAPACK_DIR};-lblas"


export ATDM_CONFIG_MPI_EXEC=jsrun

export ATDM_CONFIG_MPI_POST_FLAGS="--rs_per_socket;4"
export ATDM_CONFIG_MPI_EXEC_NUMPROCS_FLAG="-p"

export CC=$MPICC
export CXX=$MPICXX
export FC=$MPIF90
export F90=$MPIF90

module list
cat <<- EOF
Final Config for Lassen:
MPICC=$MPICC
MPICXX=$MPICXX
MPIF90=$MPIF90

CC=$CC
CXX=$CXX
FC=$FC
F90=$F90
EOF

export ATDM_CONFIG_COMPLETED_ENV_SETUP=TRUE

