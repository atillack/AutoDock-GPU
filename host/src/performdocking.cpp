/*
 * (C) 2013. Evopro Innovation Kft.
 *
 * performdocking.cu
 *
 * Created on: 2010.04.20.
 * Author: pechan.imre
 */

#define STRINGIZE2(s) #s
#define STRINGIZE(s)	STRINGIZE2(s)
#define KRNL_FILE STRINGIZE(KRNL_SOURCE)
#define KRNL_FOLDER STRINGIZE(KRNL_DIRECTORY)
#define KRNL1 STRINGIZE(K1)
#define KRNL2 STRINGIZE(K2)
#define KRNL3 STRINGIZE(K3)
#define KRNL4 STRINGIZE(K4)

#define INC " -I " KRNL_FOLDER

#if defined (N16WI)
	#define KNWI " -DN16WI "
#elif defined (N32WI)
	#define KNWI " -DN32WI "
#elif defined (N64WI)
	#define KNWI " -DN64WI "
#elif defined (N128WI)
	#define KNWI " -DN128WI "
#else
	#define KNWI	" -DN64WI "
#endif

#if defined (REPRO)
	#define REP " -DREPRO "
#else
	#define REP " "
#endif

#define KGDB_AMD 		" -g -O0 "
#define KGDB_INTEL	" -g -s " KRNL_FILE

#if defined (DOCK_DEBUG)
	#if defined (CPU_DEVICE)
		#define KGDB KGDB_INTEL
	#elif defined (GPU_DEVICE)
		#define KGDB KGDB_AMD
	#endif
#else
	#define KGDB " "
#endif

#define OPT_PROG INC KNWI REP KGDB

#include "performdocking.h"

int docking_with_gpu(const Gridinfo*   mygrid,
	             	 /*const*/ float*      cpu_floatgrids,
                           Dockpars*   mypars,
		     						 const Liganddata* myligand_init,
		     			 	     const int*        argc,
		           						 char**      argv,
		           						 clock_t     clock_start_program)
/* The function performs the docking algorithm and generates the corresponding result files.
parameter mygrid:
		describes the grid
		filled with get_gridinfo()
parameter cpu_floatgrids:
		points to the memory region containing the grids
		filled with get_gridvalues_f()
parameter mypars:
		describes the docking parameters
		filled with get_commandpars()
parameter myligand_init:
		describes the ligands
		filled with get_liganddata()
parameters argc and argv:
		are the corresponding command line arguments parameter clock_start_program:
		contains the state of the clock tick counter at the beginning of the program
filled with clock() */
{
// =======================================================================
// OpenCL Host Setup
// =======================================================================
	cl_platform_id* platform_id;
	cl_device_id* device_id;
	cl_context context;
	cl_command_queue command_queue;

	const char *filename = KRNL_FILE;
	printf("%-40s %-40s\n", "Kernel source file: ", filename);  fflush(stdout);

	const char* options_program = OPT_PROG;
	printf("%-40s %-40s\n", "Kernel compilation flags: ", options_program); fflush(stdout);

	cl_kernel kernel1; const char *name_k1 = KRNL1;
	size_t kernel1_gxsize, kernel1_lxsize;

	cl_kernel kernel2; const char *name_k2 = KRNL2;
	size_t kernel2_gxsize, kernel2_lxsize;

	cl_kernel kernel3; const char *name_k3 = KRNL3;
	size_t kernel3_gxsize, kernel3_lxsize;

	cl_kernel kernel4; const char *name_k4 = KRNL4;
	size_t kernel4_gxsize, kernel4_lxsize;

	cl_uint platformCount;
	cl_uint deviceCount;

	// Times
	cl_ulong time_start_kernel;
	cl_ulong time_end_kernel;

	// Get all available platforms
	if (getPlatforms(&platform_id,&platformCount) != 0) return 1;

	// Get all devices of first platform
	if (getDevices(platform_id[0],platformCount,&device_id,&deviceCount) != 0) return 1;

	// Create context from first platform
	if (createContext(platform_id[0],1,device_id,&context) != 0) return 1;

	// Create command queue for first device
	if (createCommandQueue(context,device_id[0],&command_queue) != 0) return 1;

	// Create program and kernel from source
	if (ImportSource(filename, name_k1, device_id, context, options_program, &kernel1) != 0) return 1;
	if (ImportSource(filename, name_k2, device_id, context, options_program, &kernel2) != 0) return 1;
	if (ImportSource(filename, name_k3, device_id, context, options_program, &kernel3) != 0) return 1;
	if (ImportSource(filename, name_k4, device_id, context, options_program, &kernel4) != 0) return 1;

// End of OpenCL Host Setup
// =======================================================================

	Liganddata myligand_reference;

	float* cpu_init_populations;
	float* cpu_final_populations;
	float* cpu_energies;
	Ligandresult* cpu_result_ligands;
	unsigned int* cpu_prng_seeds;
	int* cpu_evals_of_runs;
	float* cpu_ref_ori_angles;

	Dockparameters dockpars;
	size_t size_floatgrids;
	size_t size_populations;
	size_t size_energies;
	size_t size_prng_seeds;
	size_t size_evals_of_runs;

	int threadsPerBlock;
	int blocksPerGridForEachEntity;
	int blocksPerGridForEachRun;
	int blocksPerGridForEachLSEntity;

	unsigned long run_cnt;	/* int run_cnt; */
	int generation_cnt;
	int i;
	double progress;

	int curr_progress_cnt;
	int new_progress_cnt;

	clock_t clock_start_docking;
	clock_t	clock_stop_docking;
	clock_t clock_stop_program_before_clustering;

	//setting number of blocks and threads
	threadsPerBlock = NUM_OF_THREADS_PER_BLOCK;
	blocksPerGridForEachEntity = mypars->pop_size * mypars->num_of_runs;
	blocksPerGridForEachRun = mypars->num_of_runs;

	//allocating CPU memory for initial populations
	size_populations = mypars->num_of_runs * mypars->pop_size * GENOTYPE_LENGTH_IN_GLOBMEM*sizeof(float);
	cpu_init_populations = (float*) malloc(size_populations);
	memset(cpu_init_populations, 0, size_populations);

	//allocating CPU memory for results
	size_energies = mypars->pop_size * mypars->num_of_runs * sizeof(float);
	cpu_energies = (float*) malloc(size_energies);
	cpu_result_ligands = (Ligandresult*) malloc(sizeof(Ligandresult)*(mypars->num_of_runs));
	cpu_final_populations = cpu_init_populations;

	//allocating memory in CPU for reference orientation angles
	cpu_ref_ori_angles = (float*) malloc(mypars->num_of_runs*3*sizeof(float));

	//generating initial populations and random orientation angles of reference ligand
	//(ligand will be moved to origo and scaled as well)
	myligand_reference = *myligand_init;
	gen_initpop_and_reflig(mypars, cpu_init_populations, cpu_ref_ori_angles, &myligand_reference, mygrid);

	//allocating memory in CPU for pseudorandom number generator seeds and
	//generating them (seed for each thread during GA)
	size_prng_seeds = blocksPerGridForEachEntity * threadsPerBlock * sizeof(unsigned int);
	cpu_prng_seeds = (unsigned int*) malloc(size_prng_seeds);

	genseed(time(NULL));	//initializing seed generator

	for (i=0; i<blocksPerGridForEachEntity*threadsPerBlock; i++)
#if defined (REPRO)
		cpu_prng_seeds[i] = 1u;
#else
		cpu_prng_seeds[i] = genseed(0u);
#endif

	//allocating memory in CPU for evaluation counters
	size_evals_of_runs = mypars->num_of_runs*sizeof(int);
	cpu_evals_of_runs = (int*) malloc(size_evals_of_runs);
	memset(cpu_evals_of_runs, 0, size_evals_of_runs);

	//preparing the constant data fields for the GPU
	// ----------------------------------------------------------------------
	// The original function does CUDA calls initializing const Kernel data.
	// We create a struct to hold those constants
	// and return them <here> (<here> = where prepare_const_fields_for_gpu() is called),
	// so we can send them to Kernels from <here>, instead of from calcenergy.cpp as originally.
	// ----------------------------------------------------------------------
	// Constant struct
	kernelconstant KerConst;

	if (prepare_const_fields_for_gpu(&myligand_reference, mypars, cpu_ref_ori_angles, &KerConst) == 1)
		return 1;

	// Constant data holding struct data
	// Created because structs containing array
	// are not supported as OpenCL kernel args
  cl_mem mem_atom_charges_const;
  cl_mem mem_atom_types_const;
  cl_mem mem_intraE_contributors_const;
  cl_mem mem_VWpars_AC_const;
  cl_mem mem_VWpars_BD_const;
  cl_mem mem_dspars_S_const;
  cl_mem mem_dspars_V_const;
  cl_mem mem_rotlist_const;
  cl_mem mem_ref_coords_x_const;
  cl_mem mem_ref_coords_y_const;
  cl_mem mem_ref_coords_z_const;
  cl_mem mem_rotbonds_moving_vectors_const;
  cl_mem mem_rotbonds_unit_vectors_const;
  cl_mem mem_ref_orientation_quats_const;

	// These constants are allocated in global memory since
	// there is a limited number of constants that can be passed
	// as arguments to kernel
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_atom_charges_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(char),                          &mem_atom_types_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_INTRAE_CONTRIBUTORS*sizeof(char),                 &mem_intraE_contributors_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float),      &mem_VWpars_AC_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float),      &mem_VWpars_BD_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*sizeof(float),                        &mem_dspars_S_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATYPES*sizeof(float),                        &mem_dspars_V_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ROTATIONS*sizeof(int),                       &mem_rotlist_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_x_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_y_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,MAX_NUM_OF_ATOMS*sizeof(float),                         &mem_ref_coords_z_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_NUM_OF_ROTBONDS*sizeof(float),                    &mem_rotbonds_moving_vectors_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,3*MAX_NUM_OF_ROTBONDS*sizeof(float),                    &mem_rotbonds_unit_vectors_const);
  mallocBufferObject(context,CL_MEM_READ_ONLY,4*MAX_NUM_OF_RUNS*sizeof(float),                        &mem_ref_orientation_quats_const);

  memcopyBufferObjectToDevice(command_queue,mem_atom_charges_const,         	&KerConst.atom_charges_const,           MAX_NUM_OF_ATOMS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_atom_types_const,           	&KerConst.atom_types_const,             MAX_NUM_OF_ATOMS*sizeof(char));
  memcopyBufferObjectToDevice(command_queue,mem_intraE_contributors_const,  	&KerConst.intraE_contributors_const,    3*MAX_INTRAE_CONTRIBUTORS*sizeof(char));
  memcopyBufferObjectToDevice(command_queue,mem_VWpars_AC_const,            	&KerConst.VWpars_AC_const,              MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_VWpars_BD_const,            	&KerConst.VWpars_BD_const,              MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_dspars_S_const,             	&KerConst.dspars_S_const,               MAX_NUM_OF_ATYPES*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_dspars_V_const,             	&KerConst.dspars_V_const,               MAX_NUM_OF_ATYPES*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_rotlist_const,              	&KerConst.rotlist_const,                MAX_NUM_OF_ROTATIONS*sizeof(int));
  memcopyBufferObjectToDevice(command_queue,mem_ref_coords_x_const,         	&KerConst.ref_coords_x_const,           MAX_NUM_OF_ATOMS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_ref_coords_y_const,         	&KerConst.ref_coords_y_const,           MAX_NUM_OF_ATOMS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_ref_coords_z_const,         	&KerConst.ref_coords_z_const,           MAX_NUM_OF_ATOMS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_rotbonds_moving_vectors_const,&KerConst.rotbonds_moving_vectors_const,3*MAX_NUM_OF_ROTBONDS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_rotbonds_unit_vectors_const,  &KerConst.rotbonds_unit_vectors_const,  3*MAX_NUM_OF_ROTBONDS*sizeof(float));
  memcopyBufferObjectToDevice(command_queue,mem_ref_orientation_quats_const,  &KerConst.ref_orientation_quats_const,  4*MAX_NUM_OF_RUNS*sizeof(float));
	// ----------------------------------------------------------------------

 	//allocating GPU memory for populations, floatgirds,
	//energies, evaluation counters and random number generator states
	size_floatgrids = (sizeof(float)) * (mygrid->num_of_atypes+2) * (mygrid->size_xyz[0]) * (mygrid->size_xyz[1]) * (mygrid->size_xyz[2]);

	cl_mem mem_dockpars_fgrids;
	cl_mem mem_dockpars_conformations_current;
	cl_mem mem_dockpars_energies_current;
	cl_mem mem_dockpars_conformations_next;
	cl_mem mem_dockpars_energies_next;
	cl_mem mem_dockpars_evals_of_new_entities;
	cl_mem mem_gpu_evals_of_runs;
	cl_mem mem_dockpars_prng_states;

	mallocBufferObject(context,CL_MEM_READ_ONLY,size_floatgrids,         				&mem_dockpars_fgrids);
	mallocBufferObject(context,CL_MEM_READ_ONLY,size_populations,        				&mem_dockpars_conformations_current);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_energies,           			&mem_dockpars_energies_current);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_populations,        			&mem_dockpars_conformations_next);
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_energies,    	      			&mem_dockpars_energies_next);
	mallocBufferObject(context,CL_MEM_READ_WRITE,mypars->pop_size*mypars->num_of_runs*sizeof(int), 	&mem_dockpars_evals_of_new_entities);

	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	mallocBufferObject(context,CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR ,size_evals_of_runs,	  			&mem_gpu_evals_of_runs);
#else
	mallocBufferObject(context,CL_MEM_READ_WRITE,size_evals_of_runs,	  			&mem_gpu_evals_of_runs);
#endif
	// -------- Replacing with memory maps! ------------

	mallocBufferObject(context,CL_MEM_READ_WRITE,size_prng_seeds,  	      				&mem_dockpars_prng_states);

	memcopyBufferObjectToDevice(command_queue,mem_dockpars_fgrids,                /*(void *)*/ cpu_floatgrids,  size_floatgrids);
 	memcopyBufferObjectToDevice(command_queue,mem_dockpars_conformations_current, cpu_init_populations, 				size_populations);
	memcopyBufferObjectToDevice(command_queue,mem_gpu_evals_of_runs, 							cpu_evals_of_runs, 	 					size_evals_of_runs);
	memcopyBufferObjectToDevice(command_queue,mem_dockpars_prng_states,     			cpu_prng_seeds,      					size_prng_seeds);

	//preparing parameter struct
	dockpars.num_of_atoms  = ((char)  myligand_reference.num_of_atoms);
	dockpars.num_of_atypes = ((char)  myligand_reference.num_of_atypes);
	dockpars.num_of_intraE_contributors = ((int) myligand_reference.num_of_intraE_contributors);
	dockpars.gridsize_x    = ((char)  mygrid->size_xyz[0]);
	dockpars.gridsize_y    = ((char)  mygrid->size_xyz[1]);
	dockpars.gridsize_z    = ((char)  mygrid->size_xyz[2]);
	dockpars.grid_spacing  = ((float) mygrid->spacing);
	dockpars.rotbondlist_length = ((int) NUM_OF_THREADS_PER_BLOCK*(myligand_reference.num_of_rotcyc));
	dockpars.coeff_elec    = ((float) mypars->coeffs.scaled_AD4_coeff_elec);
	dockpars.coeff_desolv  = ((float) mypars->coeffs.AD4_coeff_desolv);
	dockpars.pop_size      = mypars->pop_size;
	dockpars.num_of_genes  = myligand_reference.num_of_rotbonds + 6;
	dockpars.tournament_rate = mypars->tournament_rate;
	dockpars.crossover_rate  = mypars->crossover_rate;
	dockpars.mutation_rate   = mypars->mutation_rate;
	dockpars.abs_max_dang    = mypars->abs_max_dang;
	dockpars.abs_max_dmov    = mypars->abs_max_dmov;
	dockpars.lsearch_rate    = mypars->lsearch_rate;
	dockpars.num_of_lsentities = (unsigned int) (mypars->lsearch_rate/100.0*mypars->pop_size + 0.5);
	dockpars.rho_lower_bound   = mypars->rho_lower_bound;
	dockpars.base_dmov_mul_sqrt3 = mypars->base_dmov_mul_sqrt3;
	dockpars.base_dang_mul_sqrt3 = mypars->base_dang_mul_sqrt3;
	dockpars.cons_limit        = (unsigned int) mypars->cons_limit;
	dockpars.max_num_of_iters  = (unsigned int) mypars->max_num_of_iters;
	dockpars.qasp = mypars->qasp;

	blocksPerGridForEachLSEntity = dockpars.num_of_lsentities*mypars->num_of_runs;

	clock_start_docking = clock();

	//print progress bar
	printf("\nExecuting docking runs:\n");
	printf("        20%%        40%%       60%%       80%%       100%%\n");
	printf("---------+---------+---------+---------+---------+\n");
	curr_progress_cnt = 0;

#ifdef DOCK_DEBUG
	// Main while-loop iterarion counter
	unsigned int ite_cnt = 0;
#endif

// Kernel1
  setKernelArg(kernel1,0, sizeof(dockpars.num_of_atoms),                  &dockpars.num_of_atoms);
  setKernelArg(kernel1,1, sizeof(dockpars.num_of_atypes),                 &dockpars.num_of_atypes);
  setKernelArg(kernel1,2, sizeof(dockpars.num_of_intraE_contributors),    &dockpars.num_of_intraE_contributors);
  setKernelArg(kernel1,3, sizeof(dockpars.gridsize_x),                    &dockpars.gridsize_x);
  setKernelArg(kernel1,4, sizeof(dockpars.gridsize_y),                    &dockpars.gridsize_y);
  setKernelArg(kernel1,5, sizeof(dockpars.gridsize_z),                    &dockpars.gridsize_z);
  setKernelArg(kernel1,6, sizeof(dockpars.grid_spacing),                  &dockpars.grid_spacing);
  setKernelArg(kernel1,7, sizeof(mem_dockpars_fgrids),                    &mem_dockpars_fgrids);
  setKernelArg(kernel1,8, sizeof(dockpars.rotbondlist_length),            &dockpars.rotbondlist_length);
  setKernelArg(kernel1,9, sizeof(dockpars.coeff_elec),                    &dockpars.coeff_elec);
  setKernelArg(kernel1,10,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
  setKernelArg(kernel1,11,sizeof(mem_dockpars_conformations_current),     &mem_dockpars_conformations_current);
  setKernelArg(kernel1,12,sizeof(mem_dockpars_energies_current),          &mem_dockpars_energies_current);
  setKernelArg(kernel1,13,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
  setKernelArg(kernel1,14,sizeof(dockpars.pop_size),                      &dockpars.pop_size);
  setKernelArg(kernel1,15,sizeof(dockpars.qasp),                          &dockpars.qasp);
	setKernelArg(kernel1,16,sizeof(mem_atom_charges_const),                 &mem_atom_charges_const);
  setKernelArg(kernel1,17,sizeof(mem_atom_types_const),                   &mem_atom_types_const);
  setKernelArg(kernel1,18,sizeof(mem_intraE_contributors_const),          &mem_intraE_contributors_const);
  setKernelArg(kernel1,19,sizeof(mem_VWpars_AC_const),                    &mem_VWpars_AC_const);
  setKernelArg(kernel1,20,sizeof(mem_VWpars_BD_const),                    &mem_VWpars_BD_const);
  setKernelArg(kernel1,21,sizeof(mem_dspars_S_const),                     &mem_dspars_S_const);
  setKernelArg(kernel1,22,sizeof(mem_dspars_V_const),                     &mem_dspars_V_const);
  setKernelArg(kernel1,23,sizeof(mem_rotlist_const),                      &mem_rotlist_const);
  setKernelArg(kernel1,24,sizeof(mem_ref_coords_x_const),                 &mem_ref_coords_x_const);
  setKernelArg(kernel1,25,sizeof(mem_ref_coords_y_const),                 &mem_ref_coords_y_const);
  setKernelArg(kernel1,26,sizeof(mem_ref_coords_z_const),                 &mem_ref_coords_z_const);
  setKernelArg(kernel1,27,sizeof(mem_rotbonds_moving_vectors_const),      &mem_rotbonds_moving_vectors_const);
  setKernelArg(kernel1,28,sizeof(mem_rotbonds_unit_vectors_const),        &mem_rotbonds_unit_vectors_const);
  setKernelArg(kernel1,29,sizeof(mem_ref_orientation_quats_const),        &mem_ref_orientation_quats_const);
	kernel1_gxsize = blocksPerGridForEachEntity * threadsPerBlock;
  kernel1_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("Kernel1: gSize: %u, lSize: %u\n", kernel1_gxsize, kernel1_lxsize); fflush(stdout);
#endif
// End of Kernel1

// Kernel2
  setKernelArg(kernel2,0,sizeof(mypars->pop_size),        								&mypars->pop_size);
  setKernelArg(kernel2,1,sizeof(mem_dockpars_evals_of_new_entities),      &mem_dockpars_evals_of_new_entities);
  setKernelArg(kernel2,2,sizeof(mem_gpu_evals_of_runs),                   &mem_gpu_evals_of_runs);
	kernel2_gxsize = blocksPerGridForEachRun * threadsPerBlock;
  kernel2_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("Kernel2: gSize: %u, lSize: %u\n", kernel2_gxsize, kernel2_lxsize); fflush(stdout);
#endif
// End of Kernel2

// Kernel4
	setKernelArg(kernel4,0, sizeof(dockpars.num_of_atoms),                 	&dockpars.num_of_atoms);
  setKernelArg(kernel4,1, sizeof(dockpars.num_of_atypes),                 &dockpars.num_of_atypes);
  setKernelArg(kernel4,2, sizeof(dockpars.num_of_intraE_contributors),    &dockpars.num_of_intraE_contributors);
  setKernelArg(kernel4,3, sizeof(dockpars.gridsize_x),                   	&dockpars.gridsize_x);
  setKernelArg(kernel4,4, sizeof(dockpars.gridsize_y),                    &dockpars.gridsize_y);
  setKernelArg(kernel4,5, sizeof(dockpars.gridsize_z),                    &dockpars.gridsize_z);
  setKernelArg(kernel4,6, sizeof(dockpars.grid_spacing),                  &dockpars.grid_spacing);
  setKernelArg(kernel4,7, sizeof(mem_dockpars_fgrids),                    &mem_dockpars_fgrids);
  setKernelArg(kernel4,8, sizeof(dockpars.rotbondlist_length),            &dockpars.rotbondlist_length);
  setKernelArg(kernel4,9, sizeof(dockpars.coeff_elec),                    &dockpars.coeff_elec);
  setKernelArg(kernel4,10,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
  setKernelArg(kernel4,11,sizeof(mem_dockpars_conformations_current),    	&mem_dockpars_conformations_current);
  setKernelArg(kernel4,12,sizeof(mem_dockpars_energies_current),          &mem_dockpars_energies_current);
  setKernelArg(kernel4,13,sizeof(mem_dockpars_conformations_next),        &mem_dockpars_conformations_next);
  setKernelArg(kernel4,14,sizeof(mem_dockpars_energies_next),             &mem_dockpars_energies_next);
  setKernelArg(kernel4,15,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
  setKernelArg(kernel4,16,sizeof(mem_dockpars_prng_states),               &mem_dockpars_prng_states);
  setKernelArg(kernel4,17,sizeof(dockpars.pop_size),                     	&dockpars.pop_size);
  setKernelArg(kernel4,18,sizeof(dockpars.num_of_genes),                 	&dockpars.num_of_genes);
  setKernelArg(kernel4,19,sizeof(dockpars.tournament_rate),               &dockpars.tournament_rate);
  setKernelArg(kernel4,20,sizeof(dockpars.crossover_rate),                &dockpars.crossover_rate);
  setKernelArg(kernel4,21,sizeof(dockpars.mutation_rate),                 &dockpars.mutation_rate);
  setKernelArg(kernel4,22,sizeof(dockpars.abs_max_dmov),                  &dockpars.abs_max_dmov);
  setKernelArg(kernel4,23,sizeof(dockpars.abs_max_dang),                  &dockpars.abs_max_dang);
  setKernelArg(kernel4,24,sizeof(dockpars.qasp),                         	&dockpars.qasp);
  setKernelArg(kernel4,25,sizeof(mem_atom_charges_const),                 &mem_atom_charges_const);
  setKernelArg(kernel4,26,sizeof(mem_atom_types_const),                  	&mem_atom_types_const);
  setKernelArg(kernel4,27,sizeof(mem_intraE_contributors_const),          &mem_intraE_contributors_const);
  setKernelArg(kernel4,28,sizeof(mem_VWpars_AC_const),                    &mem_VWpars_AC_const);
  setKernelArg(kernel4,29,sizeof(mem_VWpars_BD_const),                    &mem_VWpars_BD_const);
  setKernelArg(kernel4,30,sizeof(mem_dspars_S_const),                     &mem_dspars_S_const);
  setKernelArg(kernel4,31,sizeof(mem_dspars_V_const),                    	&mem_dspars_V_const);
  setKernelArg(kernel4,32,sizeof(mem_rotlist_const),                      &mem_rotlist_const);
  setKernelArg(kernel4,33,sizeof(mem_ref_coords_x_const),                 &mem_ref_coords_x_const);
  setKernelArg(kernel4,34,sizeof(mem_ref_coords_y_const),                 &mem_ref_coords_y_const);
  setKernelArg(kernel4,35,sizeof(mem_ref_coords_z_const),                 &mem_ref_coords_z_const);
  setKernelArg(kernel4,36,sizeof(mem_rotbonds_moving_vectors_const),     	&mem_rotbonds_moving_vectors_const);
  setKernelArg(kernel4,37,sizeof(mem_rotbonds_unit_vectors_const),        &mem_rotbonds_unit_vectors_const);
  setKernelArg(kernel4,38,sizeof(mem_ref_orientation_quats_const),       	&mem_ref_orientation_quats_const);

	kernel4_gxsize = blocksPerGridForEachEntity * threadsPerBlock;
  kernel4_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("Kernel4: gSize: %u, lSize: %u\n", kernel4_gxsize, kernel4_lxsize); fflush(stdout);
#endif
// End of Kernel4

// Kernel3
	setKernelArg(kernel3,0,sizeof(dockpars.num_of_atoms),                   &dockpars.num_of_atoms);
  setKernelArg(kernel3,1,sizeof(dockpars.num_of_atypes),                  &dockpars.num_of_atypes);
  setKernelArg(kernel3,2,sizeof(dockpars.num_of_intraE_contributors),     &dockpars.num_of_intraE_contributors);
  setKernelArg(kernel3,3,sizeof(dockpars.gridsize_x),                     &dockpars.gridsize_x);
  setKernelArg(kernel3,4,sizeof(dockpars.gridsize_y),                     &dockpars.gridsize_y);
  setKernelArg(kernel3,5,sizeof(dockpars.gridsize_z),                     &dockpars.gridsize_z);
  setKernelArg(kernel3,6,sizeof(dockpars.grid_spacing),                   &dockpars.grid_spacing);
  setKernelArg(kernel3,7,sizeof(mem_dockpars_fgrids),                     &mem_dockpars_fgrids);
  setKernelArg(kernel3,8,sizeof(dockpars.rotbondlist_length),             &dockpars.rotbondlist_length);
  setKernelArg(kernel3,9,sizeof(dockpars.coeff_elec),                     &dockpars.coeff_elec);
  setKernelArg(kernel3,10,sizeof(dockpars.coeff_desolv),                  &dockpars.coeff_desolv);
  setKernelArg(kernel3,11,sizeof(mem_dockpars_conformations_next),        &mem_dockpars_conformations_next);
  setKernelArg(kernel3,12,sizeof(mem_dockpars_energies_next),             &mem_dockpars_energies_next);
  setKernelArg(kernel3,13,sizeof(mem_dockpars_evals_of_new_entities),     &mem_dockpars_evals_of_new_entities);
  setKernelArg(kernel3,14,sizeof(mem_dockpars_prng_states),               &mem_dockpars_prng_states);
  setKernelArg(kernel3,15,sizeof(dockpars.pop_size),                      &dockpars.pop_size);
  setKernelArg(kernel3,16,sizeof(dockpars.num_of_genes),                  &dockpars.num_of_genes);
  setKernelArg(kernel3,17,sizeof(dockpars.lsearch_rate),                  &dockpars.lsearch_rate);
  setKernelArg(kernel3,18,sizeof(dockpars.num_of_lsentities),             &dockpars.num_of_lsentities);
  setKernelArg(kernel3,19,sizeof(dockpars.rho_lower_bound),               &dockpars.rho_lower_bound);
  setKernelArg(kernel3,20,sizeof(dockpars.base_dmov_mul_sqrt3),           &dockpars.base_dmov_mul_sqrt3);
  setKernelArg(kernel3,21,sizeof(dockpars.base_dang_mul_sqrt3),           &dockpars.base_dang_mul_sqrt3);
  setKernelArg(kernel3,22,sizeof(dockpars.cons_limit),                    &dockpars.cons_limit);
  setKernelArg(kernel3,23,sizeof(dockpars.max_num_of_iters),              &dockpars.max_num_of_iters);
  setKernelArg(kernel3,24,sizeof(dockpars.qasp),                          &dockpars.qasp);
  setKernelArg(kernel3,25,sizeof(mem_atom_charges_const),                 &mem_atom_charges_const);
  setKernelArg(kernel3,26,sizeof(mem_atom_types_const),                   &mem_atom_types_const);
  setKernelArg(kernel3,27,sizeof(mem_intraE_contributors_const),          &mem_intraE_contributors_const);
  setKernelArg(kernel3,28,sizeof(mem_VWpars_AC_const),                    &mem_VWpars_AC_const);
  setKernelArg(kernel3,29,sizeof(mem_VWpars_BD_const),                    &mem_VWpars_BD_const);
  setKernelArg(kernel3,30,sizeof(mem_dspars_S_const),                     &mem_dspars_S_const);
  setKernelArg(kernel3,31,sizeof(mem_dspars_V_const),                     &mem_dspars_V_const);
  setKernelArg(kernel3,32,sizeof(mem_rotlist_const),                      &mem_rotlist_const);
  setKernelArg(kernel3,33,sizeof(mem_ref_coords_x_const),                 &mem_ref_coords_x_const);
  setKernelArg(kernel3,34,sizeof(mem_ref_coords_y_const),                 &mem_ref_coords_y_const);
  setKernelArg(kernel3,35,sizeof(mem_ref_coords_z_const),                 &mem_ref_coords_z_const);
  setKernelArg(kernel3,36,sizeof(mem_rotbonds_moving_vectors_const),      &mem_rotbonds_moving_vectors_const);
  setKernelArg(kernel3,37,sizeof(mem_rotbonds_unit_vectors_const),        &mem_rotbonds_unit_vectors_const);
  setKernelArg(kernel3,38,sizeof(mem_ref_orientation_quats_const),        &mem_ref_orientation_quats_const);
  kernel3_gxsize = blocksPerGridForEachLSEntity * threadsPerBlock;
  kernel3_lxsize = threadsPerBlock;
#ifdef DOCK_DEBUG
	printf("Kernel3: gSize: %u, lSize: %u\n", kernel3_gxsize, kernel3_lxsize); fflush(stdout);
#endif
// End of Kernel3


// Kernel1
	#ifdef DOCK_DEBUG
		printf("Start Kernel1 ... ");fflush(stdout);
	#endif
	runKernel1D(command_queue,kernel1,kernel1_gxsize,kernel1_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf(" ... Finish Kernel1\n");fflush(stdout);
	#endif
// End of Kernel1

// Kernel2
	#ifdef DOCK_DEBUG
		printf("Start Kernel2 ... ");fflush(stdout);
	#endif
	runKernel1D(command_queue,kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf(" ... Finish Kernel2\n");fflush(stdout);
	#endif
// End of Kernel2
	// ===============================================================================


	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	int* map_cpu_evals_of_runs;
	map_cpu_evals_of_runs = (int*) memMap(command_queue, mem_gpu_evals_of_runs, CL_MAP_READ, size_evals_of_runs);
#else
	memcopyBufferObjectFromDevice(command_queue,cpu_evals_of_runs,mem_gpu_evals_of_runs,size_evals_of_runs);
#endif
	// -------- Replacing with memory maps! ------------


	generation_cnt = 1;


	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	while ((progress = check_progress(map_cpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs)) < 100.0)
#else
	while ((progress = check_progress(cpu_evals_of_runs, generation_cnt, mypars->num_of_energy_evals, mypars->num_of_generations, mypars->num_of_runs)) < 100.0)
#endif
	// -------- Replacing with memory maps! ------------

	{
#ifdef DOCK_DEBUG
    ite_cnt++;
    printf("Iteration # %u\n", ite_cnt);
    fflush(stdout);
#endif

	 //update progress bar (bar length is 50)
	 new_progress_cnt = (int) (progress/2.0+0.5);
	 if (new_progress_cnt > 50)
	 	new_progress_cnt = 50;

	 while (curr_progress_cnt < new_progress_cnt) {
		curr_progress_cnt++;
		printf("*");
		fflush(stdout);
	}

// Kernel4
	#ifdef DOCK_DEBUG
		printf("Start Kernel4 ... ");fflush(stdout);
	#endif
		runKernel1D(command_queue,kernel4,kernel4_gxsize,kernel4_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf(" ... Finish Kernel4\n");fflush(stdout);
	#endif
// End of Kernel4

// Kernel3
	#ifdef DOCK_DEBUG
		printf("Start Kernel3 ... ");fflush(stdout);
	#endif
		runKernel1D(command_queue,kernel3,kernel3_gxsize,kernel3_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf(" ... Finish Kernel3\n");fflush(stdout);
	#endif
// End of Kernel3

// Kernel2
	#ifdef DOCK_DEBUG
		printf("Start Kernel2 ... ");fflush(stdout);
	#endif
		runKernel1D(command_queue,kernel2,kernel2_gxsize,kernel2_lxsize,&time_start_kernel,&time_end_kernel);
	#ifdef DOCK_DEBUG
		printf(" ... Finish Kernel2\n");fflush(stdout);
	#endif
// End of Kernel2
		// ===============================================================================


		// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
		map_cpu_evals_of_runs = (int*) memMap(command_queue, mem_gpu_evals_of_runs, CL_MAP_READ, size_evals_of_runs);
#else
		memcopyBufferObjectFromDevice(command_queue,cpu_evals_of_runs,mem_gpu_evals_of_runs,size_evals_of_runs);
#endif
		// -------- Replacing with memory maps! ------------

		generation_cnt++;

		// ----------------------------------------------------------------------
		// ORIGINAL APPROACH: switching conformation and energy pointers
		// CURRENT APPROACH:  copy data from one buffer to another, pointers are kept the same
		// IMPROVED CURRENT APPROACH
		// Kernel arguments are changed on every iteration
		// No copy from dev glob memory to dev glob memory occurs
		// Use generation_cnt as it evolves with the main loop
		// No need to use tempfloat
		// No performance improvement wrt to "CURRENT APPROACH"

		// Kernel args exchange regions they point to
		// But never two args point to the same region of dev memory
		// NO ALIASING -> use restrict in Kernel
		if (generation_cnt % 2 == 0) {
			// Kernel 4
			setKernelArg(kernel4,11,sizeof(mem_dockpars_conformations_next),                &mem_dockpars_conformations_next);
			setKernelArg(kernel4,12,sizeof(mem_dockpars_energies_next),                     &mem_dockpars_energies_next);
      setKernelArg(kernel4,13,sizeof(mem_dockpars_conformations_current),             &mem_dockpars_conformations_current);
			setKernelArg(kernel4,14,sizeof(mem_dockpars_energies_current),                  &mem_dockpars_energies_current);

			// Kernel 3
     	setKernelArg(kernel3,11,sizeof(mem_dockpars_conformations_current),             &mem_dockpars_conformations_current);
      setKernelArg(kernel3,12,sizeof(mem_dockpars_energies_current),                  &mem_dockpars_energies_current);
		}
		else { // In this configuration, the program starts
			// Kernel 4
			setKernelArg(kernel4,11,sizeof(mem_dockpars_conformations_current),             &mem_dockpars_conformations_current);
			setKernelArg(kernel4,12,sizeof(mem_dockpars_energies_current),                  &mem_dockpars_energies_current);
      setKernelArg(kernel4,13,sizeof(mem_dockpars_conformations_next),                &mem_dockpars_conformations_next);
			setKernelArg(kernel4,14,sizeof(mem_dockpars_energies_next),                     &mem_dockpars_energies_next);

			// Kernel 3
			setKernelArg(kernel3,11,sizeof(mem_dockpars_conformations_next),                &mem_dockpars_conformations_next);
      setKernelArg(kernel3,12,sizeof(mem_dockpars_energies_next),                     &mem_dockpars_energies_next);
		}
		// ----------------------------------------------------------------------

#ifdef DOCK_DEBUG
        printf("Progress %.3f %%\n", progress);
        fflush(stdout);
#endif
	} // End of while-loop


	// -------- Replacing with memory maps! ------------
#if defined (MAPPED_COPY)
	unmemMap(command_queue,mem_gpu_evals_of_runs,map_cpu_evals_of_runs);
#endif
	// -------- Replacing with memory maps! ------------


	clock_stop_docking = clock();

	//update progress bar (bar length is 50)
	while (curr_progress_cnt < 50) {
		curr_progress_cnt++;
		printf("*");
		fflush(stdout);
	}

	printf("\n\n");

	// ===============================================================================
	// L30nardoSV modified
	// http://www.cc.gatech.edu/~vetter/keeneland/tutorial-2012-02-20/08-opencl.pdf
	// ===============================================================================

	//processing results
	memcopyBufferObjectFromDevice(command_queue,cpu_final_populations,mem_dockpars_conformations_current,size_populations);
	memcopyBufferObjectFromDevice(command_queue,cpu_energies,mem_dockpars_energies_current,size_energies);

#if defined (DOCK_DEBUG)
	for (int cnt_pop=0;cnt_pop<size_populations/sizeof(float);cnt_pop++)
		printf("total_num_pop: %u, cpu_final_populations[%u]: %f\n",(unsigned int)(size_populations/sizeof(float)),cnt_pop,cpu_final_populations[cnt_pop]);

	for (int cnt_pop=0;cnt_pop<size_energies/sizeof(float);cnt_pop++)
		printf("total_num_energies: %u, cpu_energies[%u]: %f\n",    (unsigned int)(size_energies/sizeof(float)),cnt_pop,cpu_energies[cnt_pop]);
#endif

	// ===============================================================================


	for (run_cnt=0; run_cnt < mypars->num_of_runs; run_cnt++)
	{
		arrange_result(cpu_final_populations+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, cpu_energies+run_cnt*mypars->pop_size, mypars->pop_size);

		make_resfiles(cpu_final_populations+run_cnt*mypars->pop_size*GENOTYPE_LENGTH_IN_GLOBMEM, cpu_energies+run_cnt*mypars->pop_size, &myligand_reference,
					  myligand_init, mypars, cpu_evals_of_runs[run_cnt], generation_cnt, mygrid, cpu_floatgrids, cpu_ref_ori_angles+3*run_cnt, argc, argv, /*1*/0,
					  run_cnt, &(cpu_result_ligands [run_cnt]));

	}

	clock_stop_program_before_clustering = clock();
	clusanal_gendlg(cpu_result_ligands, mypars->num_of_runs, myligand_init, mypars,
					 mygrid, argc, argv, ELAPSEDSECS(clock_stop_docking, clock_start_docking)/mypars->num_of_runs,
					 ELAPSEDSECS(clock_stop_program_before_clustering, clock_start_program));


	clock_stop_docking = clock();

	clReleaseMemObject(mem_atom_charges_const);
  clReleaseMemObject(mem_atom_types_const);
  clReleaseMemObject(mem_intraE_contributors_const);
  clReleaseMemObject(mem_VWpars_AC_const);
	clReleaseMemObject(mem_VWpars_BD_const);
	clReleaseMemObject(mem_dspars_S_const);
	clReleaseMemObject(mem_dspars_V_const);
  clReleaseMemObject(mem_rotlist_const);
	clReleaseMemObject(mem_ref_coords_x_const);
	clReleaseMemObject(mem_ref_coords_y_const);
	clReleaseMemObject(mem_ref_coords_z_const);
	clReleaseMemObject(mem_rotbonds_moving_vectors_const);
	clReleaseMemObject(mem_rotbonds_unit_vectors_const);
	clReleaseMemObject(mem_ref_orientation_quats_const);

	clReleaseMemObject(mem_dockpars_fgrids);
	clReleaseMemObject(mem_dockpars_conformations_current);
	clReleaseMemObject(mem_dockpars_energies_current);
	clReleaseMemObject(mem_dockpars_conformations_next);
	clReleaseMemObject(mem_dockpars_energies_next);
	clReleaseMemObject(mem_dockpars_evals_of_new_entities);
	clReleaseMemObject(mem_dockpars_prng_states);
	clReleaseMemObject(mem_gpu_evals_of_runs);

	clReleaseKernel(kernel1);
	clReleaseKernel(kernel2);
	clReleaseKernel(kernel3);
	clReleaseKernel(kernel4);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(device_id);
	free(platform_id);

	free(cpu_init_populations);
	free(cpu_energies);
	free(cpu_result_ligands);
	free(cpu_prng_seeds);
	free(cpu_evals_of_runs);
  // -------- Replacing with memory maps! ------------
  //free(map_cpu_evals_of_runs);
	// -------- Replacing with memory maps! ------------
	free(cpu_ref_ori_angles);

	return 0;
}

double check_progress(int* evals_of_runs, int generation_cnt, int max_num_of_evals, int max_num_of_gens, int num_of_runs)
//The function checks if the stop condition of the docking is satisfied, returns 0 if no, and returns 1 if yes. The fitst
//parameter points to the array which stores the number of evaluations performed for each run. The second parameter stores
//the generations used. The other parameters describe the maximum number of energy evaluations, the maximum number of
//generations, and the number of runs, respectively. The stop condition is satisfied, if the generations used is higher
//than the maximal value, or if the average number of evaluations used is higher than the maximal value.
{
	/*	Stops if every run reached the number of evals or number of generations

	int runs_finished;
	int i;

	runs_finished = 0;
	for (i=0; i<num_of_runs; i++)
		if (evals_of_runs[i] >= max_num_of_evals)
			runs_finished++;

	if ((runs_finished >= num_of_runs) || (generation_cnt >= max_num_of_gens))
		return 1;
	else
		return 0;
        */

	//Stops if the sum of evals of every run reached the sum of the total number of evals

	double total_evals;
	int i;
	double evals_progress;
	double gens_progress;

	//calculating progress according to number of runs
	total_evals = 0.0;
	for (i=0; i<num_of_runs; i++)
		total_evals += evals_of_runs[i];

	evals_progress = total_evals/((double) num_of_runs)/max_num_of_evals*100.0;

	//calculating progress according to number of generations
	gens_progress = ((double) generation_cnt)/((double) max_num_of_gens)*100.0;

	if (evals_progress > gens_progress)
		return evals_progress;
	else
		return gens_progress;
}