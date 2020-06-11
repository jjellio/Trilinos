C    Copyright(C) 1999-2020 National Technology & Engineering Solutions
C    of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
C    NTESS, the U.S. Government retains certain rights in this software.
C    
C    See packages/seacas/LICENSE for details
      SUBROUTINE MXCOMP (MYV, VOID, LVOID,
     *   NVOIDS, DPOINT, LDICT, NNAMES, CHRCOL, LASTER)
C
      IMPLICIT INTEGER (A-Z)
      INCLUDE 'params.inc'
C
C     THIS ROUTINE PERFORMS THE NUMERIC DATA COMPRESSION OPERATION.
C
C************************************************************************
C
C     MYV      Reference array
C     VOID     Void table
C     LVOID    Dimension of VOID
C     NVOIDS   Number of voids
C     DPOINT   Dictionary pointer table
C     LDICT    Dimension of DPOINT
C     NNAMES   Number of names
C     CHRCOL   Column for character tables
C     LASTER   Error return code
C
      DIMENSION DPOINT(LDICT,CHRCOL,2), VOID(LVOID,CHRCOL,2), MYV(*)
      DIMENSION NNAMES(2), NVOIDS(2)
C
C************************************************************************
C
      LASTER = SUCESS
C
C     The basic strategy is to look for an array in the dictionary
C     which is immediately preceded by a void.  If found, a shift
C     is performed, and the void table is updated.
C
      IVOID = 0
  100 CONTINUE
      IVOID = IVOID + 1
  110 IF (IVOID .GT. NVOIDS(1)) GO TO 130
         VADDR = VOID(IVOID,1,1) + VOID(IVOID,1,2)
         DO 120 IDICT = 1, NNAMES(1)
            DADDR = DPOINT(IDICT,1,1)
            IF (VADDR .EQ. DADDR .AND. DPOINT(IDICT,1,2) .GT. 0) THEN
C
C              Perform data shift and update void table.
C
               CALL SHFTI (MYV, 1, 1,
     *            DADDR, DADDR+DPOINT(IDICT,1,2)-1, VOID(IVOID,1,2))
               DPOINT(IDICT,1,1) = VOID(IVOID,1,1)
               VOID(IVOID,1,1) = DPOINT(IDICT,1,1) + DPOINT(IDICT,1,2)
               CALL VTABLE (0, 0, VOID, LVOID, NVOIDS(1), CHRCOL,
     *            LASTER)
               IF (LASTER .NE. SUCESS) RETURN
               GO TO 110
C
            END IF
  120    CONTINUE
      GO TO 100
  130 CONTINUE
      RETURN
      END
