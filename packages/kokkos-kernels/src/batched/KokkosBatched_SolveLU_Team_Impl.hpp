#ifndef __KOKKOSBATCHED_SOLVELU_TEAM_IMPL_HPP__
#define __KOKKOSBATCHED_SOLVELU_TEAM_IMPL_HPP__


/// \author Vinh Dang (vqdang@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"

namespace KokkosBatched {
  namespace Experimental {
    ///
    /// Team Impl
    /// =========

    ///
    /// SolveLU no piv
    ///
    
    template<typename MemberType>
    struct TeamSolveLU<MemberType,Algo::SolveLU::Unblocked,Trans::NoTranspose> {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const BViewType &B) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert((BViewType::rank == 1)||(BViewType::rank == 2), "B should have either one dimension or two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename BViewType::memory_space>::value, "A and B should be on the same memory space");
        assert(A.extent(0)==A.extent(1));
        assert(A.extent(1)==B.extent(0));

        typedef typename AViewType::value_type ScalarType;

        const ScalarType one(1.0);

        //First, compute Y (= U*X) by solving the system L*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);
        //Second, compute X by solving the system U*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);

        return 0;
      }
    };
    
    template<typename MemberType>
    struct TeamSolveLU<MemberType,Algo::SolveLU::Blocked,Trans::NoTranspose> {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const BViewType &B) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert((BViewType::rank == 1)||(BViewType::rank == 2), "B should have either one dimension or two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename BViewType::memory_space>::value, "A and B should be on the same memory space");
        assert(A.extent(0)==A.extent(1));
        assert(A.extent(1)==B.extent(0));

        typedef typename AViewType::value_type ScalarType;

        const ScalarType one(1.0);

        //First, compute Y (= U*X) by solving the system L*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit,Algo::Trsm::Blocked>::invoke(member, one, A, B);
        //Second, compute X by solving the system U*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(member, one, A, B);

        return 0;
      }
    };
	
    template<typename MemberType>
    struct TeamSolveLU<MemberType,Algo::SolveLU::Unblocked,Trans::Transpose> {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const BViewType &B) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert((BViewType::rank == 1)||(BViewType::rank == 2), "B should have either one dimension or two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename BViewType::memory_space>::value, "A and B should be on the same memory space");
        assert(A.extent(0)==A.extent(1));
        assert(A.extent(1)==B.extent(0));

        typedef typename AViewType::value_type ScalarType;

        const ScalarType one(1.0);

        //First, compute Y (= L'*X) by solving the system U'*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::Transpose,Diag::NonUnit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);
        //Second, compute X by solving the system L'*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::Transpose,Diag::Unit,Algo::Trsm::Unblocked>::invoke(member, one, A, B);

        return 0;
      }
    };

    template<typename MemberType>
    struct TeamSolveLU<MemberType,Algo::SolveLU::Blocked,Trans::Transpose> {
      template<typename AViewType,
               typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType &member, const AViewType &A, const BViewType &B) {
        static_assert(AViewType::rank == 2, "A should have two dimensions");
        static_assert((BViewType::rank == 1)||(BViewType::rank == 2), "B should have either one dimension or two dimensions");
        static_assert(std::is_same<typename AViewType::memory_space, typename BViewType::memory_space>::value, "A and B should be on the same memory space");
        assert(A.extent(0)==A.extent(1));
        assert(A.extent(1)==B.extent(0));

        typedef typename AViewType::value_type ScalarType;

        const ScalarType one(1.0);

        //First, compute Y (= L'*X) by solving the system U'*Y = B for Y
        TeamTrsm<MemberType,Side::Left,Uplo::Lower,Trans::Transpose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(member, one, A, B);
        //Second, compute X by solving the system L'*X = Y for X
        TeamTrsm<MemberType,Side::Left,Uplo::Upper,Trans::Transpose,Diag::Unit,Algo::Trsm::Blocked>::invoke(member, one, A, B);

        return 0;
      }
    };

  }
}

#endif
