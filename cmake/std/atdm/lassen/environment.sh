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

module purge --silent
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


# Prepend path to ninja after all of the modules are loaded
export PATH=/usr/workspace/emplasma/TPLs/bin/:$PATH

# Set a standard git so everyone has the same git
module load git/2.20.0

# ATDM specific config variables
export ATDM_CONFIG_LAPACK_LIBS="-L${LAPACK_DIR};-llapack"
export ATDM_CONFIG_BLAS_LIBS="-L${LAPACK_DIR};-lblas"

# Set common MPI wrappers
export MPICC=$MVAPICH2_ORIG_MPICC
export MPICXX=$MVAPICH2_ORIG_MPICXX
export MPIF90=$MVAPICH2_ORIG_MPIFC

export ATDM_CONFIG_MPI_EXEC=jsrun

export ATDM_CONFIG_MPI_POST_FLAGS="--rs_per_socket;4"
export ATDM_CONFIG_MPI_EXEC_NUMPROCS_FLAG="-p"

cat <<- EOF
Final Config for Lassen:
MPICC=$MPICC
MPICXX=$MPICXX
MPIF90=$MPIF90
EOF

export ATDM_CONFIG_COMPLETED_ENV_SETUP=TRUE

