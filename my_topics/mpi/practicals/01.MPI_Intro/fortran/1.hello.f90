PROGRAM hello

!==============================================================!
!                                                              !
! This file has been written as a sample solution to an        !
! exercise in a course given at the CSCS Summer School.        !
! It is made freely available with the understanding that      !
! every copy of this file must include this header and that    !
! CSCS take no responsibility for the use of the enclosed      !
! teaching material.                                           !
!                                                              !
! Purpose: a simple MPI-program printing "hello world!"        !
!                                                              !
! Contents: F-Source                                           !
!==============================================================!

! Write a minimal  MPI program which prints "hello world by each MPI process

! Include header file
  use mpi

  IMPLICIT NONE

  ! Initialize MPI
  integer :: ierr, i
  integer :: myid, ncpu

  call MPI_INIT(ierr)

  ! Print hello world from each process
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, ncpu, ierr)

  do i=0, ncpu-1
    if (myid==i) write(*,*) "Hello from rank", myid
    call MPI_BARRIER(MPI_COMM_WORLD, ierr)
  enddo

  ! Finalize MPI
  call MPI_FINALIZE(ierr)

END PROGRAM
