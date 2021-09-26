/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr,int rank,int size, int* nb_rows_per_rank , int* offset_index, float**  av_vels_per_rank);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int rows,int offset,int rank);
float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles,int start ,int rows, int cols,int rank,int offset,float c_sq,float w0,float w1,float w2);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, char* mode,int rows,int offset);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells,int rows,int cols);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles,int rows, int cols,int offset,int tot);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles,int rows,int cols,int offset,int denom);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

int synchronized_writing(const t_param params, t_speed* cells, int* obstacles, float* av_vels,int rows,int rank,int size,int offset);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
 

	int sizes,ranks ;
        int size = 0;
        int rank = 0;
        int cols = 0 ;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &sizes );
  size = sizes;
  MPI_Comm_rank( MPI_COMM_WORLD, &ranks );
  rank = ranks;
  
  int nb_rows_per_rank[112];
  int offset_index[112];
  float* av_vels_per_rank = NULL ;
  int tot_cells ;
  float inv ;

  /* Total/init time starts here: initialise our data structures and load values from file */
  tot_cells = initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels,rank,size,nb_rows_per_rank,offset_index, &av_vels_per_rank);
  inv = 1.f/tot_cells;
  MPI_Datatype unit_cell;
  MPI_Datatype cells_row;
  MPI_Type_contiguous(9, MPI_FLOAT, &unit_cell);
  MPI_Type_commit(&unit_cell);
  MPI_Type_contiguous(params.nx,unit_cell,&cells_row);
  MPI_Type_commit(&cells_row);
  cols = params.nx;

  /* Init time stops here, compute time starts*/


  int up,down;
  up = rank == 0 ? size - 1 : rank -1 ;
  down = rank == size - 1 ? 0 : rank +1 ;
  int myRank_actual = nb_rows_per_rank[rank];
  int first,first_dos;
  first = cols*myRank_actual;
  first_dos =cols*(myRank_actual+1);  
  int myOffset_actual = offset_index[rank];
  const float c_sq = 1.f / 3.f; 
  const float w0 = 4.f / 9.f; 
  const float w1 = 1.f / 9.f;  
  const float w2 = 1.f / 36.f; 
  
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
	  
	if((rank == size-1 && myRank_actual >1) || (rank == size -2 && myRank_actual == 1)){
      accelerate_flow(params, cells, obstacles,myRank_actual,myOffset_actual,rank);
    }
  
    MPI_Sendrecv(&cells[first], 1, cells_row,down, 0, &cells[0], 1, cells_row,up , 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  	MPI_Sendrecv(&cells[cols], 1, cells_row, up , 0, &cells[first_dos], 1, cells_row,down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
    
    float u_sum = collision(params, cells, tmp_cells, obstacles,1,myRank_actual+1,cols,rank,myOffset_actual,c_sq,w0,w1,w2);
    av_vels_per_rank[tt] = u_sum*inv;
    
    t_speed * tmp2 = cells;
    cells = tmp_cells;
    tmp_cells=tmp2;
  
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells,nb_rows_per_rank[rank],cols));
#endif
  }
  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  
  MPI_Reduce(av_vels_per_rank, av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  float tmp = calc_reynolds(params, cells, obstacles,nb_rows_per_rank[rank],cols,offset_index[rank],tot_cells);
  
  /* write final values and free memory */
  if(rank ==0){
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n",tmp );
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim); 
}
  synchronized_writing(params,cells,obstacles,av_vels,nb_rows_per_rank[rank],rank,size,offset_index[rank]);
  //finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

int synchronized_writing(const t_param params, t_speed* cells, int* obstacles, float* av_vels,int rows, int rank,int size,int offset){
	
	if (rank ==0) {
	// create the file and write rank 0 ;
		write_values(params,cells,obstacles,av_vels,"w",rows,offset);
	}

	int counter = 1 ; // still havent wrote !
	int has_wrote = 0 ;
while(counter < size){
	MPI_Barrier(MPI_COMM_WORLD);
	if(counter == rank && !has_wrote && rank != 0){
		write_values(params,cells,obstacles,av_vels,"a",rows,offset);
		++ has_wrote;
	}
	++ counter ;		
}// fin du while

if(rank != 0 ){
	return EXIT_SUCCESS;
}
else {
	// deuxieme partie du write
	FILE* fp ;
	
  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);
}
	return EXIT_SUCCESS;
}


int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, int rows,int offset,int rank)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = rows - 1 < 1 ? 1 : rows-1; 
  
  //#pragma vector aligned
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
        && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
        && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[ii + jj*params.nx].speeds[1] += w1;
      cells[ii + jj*params.nx].speeds[5] += w2;
      cells[ii + jj*params.nx].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii + jj*params.nx].speeds[3] -= w1;
      cells[ii + jj*params.nx].speeds[6] -= w2;
      cells[ii + jj*params.nx].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}



float collision(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles,int start ,int rows, int cols,int rank,int offset,float c_sq,float w0,float w1,float w2)
{

   float tot_u = 0.f;
   


  for (int jj = start; jj < rows; jj++) {
	 const int y_n = (jj + 1) ;
	 const int y_s = (jj - 1); 
    #pragma omp simd
    #pragma vector aligned
    for (int ii = 0; ii < cols; ii++) {       
       const int x_e = (ii + 1) % cols;
       const int x_w = (ii == 0) ? (ii + cols - 1) : (ii - 1);
   		const float s0 = cells[ii + jj*cols].speeds[0];
        const float s1 = cells[x_w + jj*cols].speeds[1];
        const float s2 = cells[ii + y_s*cols].speeds[2];
        const float s3 = cells[x_e + jj*cols].speeds[3];
        const float s4 = cells[ii + y_n*cols].speeds[4];
        const float s5 = cells[x_w + y_s*cols].speeds[5];
        const float s6 = cells[x_e + y_s*cols].speeds[6];
        const float s7 = cells[x_e + y_n*cols].speeds[7];
        const float s8 = cells[x_w + y_n*cols].speeds[8];
       

      
      float local_density = 0.f;
      local_density=s0+s1+s2+s3+s4+s5+s6+s7+s8;
      const float inverse = 1/local_density;

    
      const float u_x = (s1+s5+s8-(s3+s6+s7))*inverse;
      
     
      const float u_y = (s2+s5+s6-(s4+s7+s8))*inverse;
     
      
       
        float u_sq = u_x * u_x + u_y * u_y;

        
        const float c_sq_sq_inv = 1/(2.f * c_sq * c_sq);
        const float c_cs_db_inv = 1/(2.f * c_sq);
        const float c_sq_inv = 1/c_sq;
       
        const float d_equ0 = w0 * local_density
                   * (1.f - u_sq *c_cs_db_inv);
      
         const float d_equ1 = w1 * local_density * (1.f + u_x *c_sq_inv
                                         + (u_x * u_x) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ2 = w1 * local_density * (1.f +  u_y *c_sq_inv
                                         + ( u_y *  u_y) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ3 = w1 * local_density * (1.f - u_x *c_sq_inv
                                         + (u_x * u_x) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ4 = w1 * local_density * (1.f - u_y *c_sq_inv
                                         + (u_y * u_y) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
        
         const float d_equ5 = w2 * local_density * (1.f + (u_x + u_y) *c_sq_inv
                                         + ((u_x + u_y) * (u_x + u_y)) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ6 = w2 * local_density * (1.f + (- u_x + u_y) *c_sq_inv
                                         + ((- u_x + u_y) * (- u_x + u_y)) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ7 = w2 * local_density * (1.f + (- u_x - u_y) *c_sq_inv
                                         + ((- u_x - u_y) * (- u_x - u_y)) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
         const float d_equ8 = w2 * local_density * (1.f + (u_x - u_y) *c_sq_inv
                                         + ((u_x - u_y) * (u_x - u_y)) * c_sq_sq_inv
                                         - u_sq *c_cs_db_inv);
        const float obs = obstacles[jj*params.nx + ii];
                                         

       
        tmp_cells[ii + jj*cols].speeds[0] = s0*obs + (1-obs)*(s0*(1-params.omega)+params.omega*d_equ0);                                   
        tmp_cells[ii + jj*cols].speeds[1] = s3*obs + (1-obs)*(s1*(1-params.omega)+params.omega*d_equ1);
        tmp_cells[ii + jj*cols].speeds[2]= s4*obs + (1-obs)*(s2*(1-params.omega)+params.omega*d_equ2);
        tmp_cells[ii + jj*cols].speeds[3]= s1*obs + (1-obs)*(s3*(1-params.omega)+params.omega*d_equ3);
        tmp_cells[ii + jj*cols].speeds[4]= s2*obs + (1-obs)*(s4*(1-params.omega)+params.omega*d_equ4);
        tmp_cells[ii + jj*cols].speeds[5]= s7*obs + (1-obs)*(s5*(1-params.omega)+params.omega*d_equ5);
        tmp_cells[ii + jj*cols].speeds[6]= s8*obs + (1-obs)*(s6*(1-params.omega)+params.omega*d_equ6);
        tmp_cells[ii + jj*cols].speeds[7]= s5*obs + (1-obs)*(s7*(1-params.omega)+params.omega*d_equ7);
        tmp_cells[ii + jj*cols].speeds[8]= s6*obs + (1-obs)*(s8*(1-params.omega)+params.omega*d_equ8);
       /*
        union
		{
			int i;
			float x;
		} u;

		u.x = (u_x * u_x) + (u_y * u_y);
		u.i = (1<<29) + (u.i >> 1) - (1<<22); 
  
        
        
        tot_u += (1-obs)*u.x;
        */
        //tot_cells += (1-obs);    
        tot_u += (1-obs)*sqrtf((u_x * u_x) + (u_y * u_y));
    }
  }
  
  return tot_u;
}




float av_velocity(const t_param params, t_speed* cells, int* obstacles,int rows, int cols,int offset,int tot)
{
  //int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */
	float output = 0 ;
	float tmp = 0 ;

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 1; jj < rows+1 ; jj++)
  {
    for (int ii = 0; ii < cols; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + (jj)*cols])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*cols].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (cells[ii + jj*cols].speeds[1]
                      + cells[ii + jj*cols].speeds[5]
                      + cells[ii + jj*cols].speeds[8]
                      - (cells[ii + jj*cols].speeds[3]
                         + cells[ii + jj*cols].speeds[6]
                         + cells[ii + jj*cols].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[ii + jj*cols].speeds[2]
                      + cells[ii + jj*cols].speeds[5]
                      + cells[ii + jj*cols].speeds[6]
                      - (cells[ii + jj*cols].speeds[4]
                         + cells[ii + jj*cols].speeds[7]
                         + cells[ii + jj*cols].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++tot_cells;
      }
    }
  }

  tmp =  tot_u/tot ;
  MPI_Reduce(&tmp, &output, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  return output;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr,int rank,int size, int* nb_rows_per_rank , int* offset_index, float**  av_vels_per_rank)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */
  int obs;
  
    
  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */




  int rows,cols,remainder;
  rows = params->ny / size;
  remainder = params->ny % size; 
  cols = params->nx;
  
  
  offset_index[0]=0;
  for(int i=0 ; i<size ; ++i){
  	// at least 3 rows per rank
  	if(remainder > 0){
  			nb_rows_per_rank[i] =  rows + 1 ;
  			--remainder;
  	}
  	else {
  		nb_rows_per_rank[i] = rows  ;
  	}
  }
  /*
  if(nb_rows_per_rank[size-1]<3){
	  
	  
	  if(nb_rows_per_rank[size-1] == 1){
		   nb_rows_per_rank[size-2] = 0;
		 //  nb_rows_per_rank[size-3] = 0;
	  }
	  if(nb_rows_per_rank[size-1] ==2){
		  nb_rows_per_rank[size-2] -= 1;
		 }
		 
	  
	  nb_rows_per_rank[size-1] = 2;
	  
  }
  */
  for(int i=1 ; i<size;++i){	    	
  		offset_index[i] = offset_index[i-1] + nb_rows_per_rank[i-1];
  }

  /* main grid */
  *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * ((nb_rows_per_rank[rank]+2) * cols),64);

  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * ((nb_rows_per_rank[rank]+2) * cols),64);

  if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (nb_rows_per_rank[rank]+2)*params->nx,64);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < nb_rows_per_rank[rank]; jj++)
  {
    for (int ii = 0; ii < cols; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + (jj+1)*cols].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + (jj+1)*cols].speeds[1] = w1;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[2] = w1;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[3] = w1;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + (jj+1)*cols].speeds[5] = w2;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[6] = w2;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[7] = w2;
      (*cells_ptr)[ii + (jj+1)*cols].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  
  for (int jj = 0; jj < nb_rows_per_rank[rank] ; jj++)
  {
    for (int ii = 0; ii < cols; ii++)
    {
      (*obstacles_ptr)[ii + (jj+1)*cols] = 0;
    }
  }
  int* full_grid = NULL;
  if(rank==0)
  {
    full_grid = (int*) _mm_malloc(sizeof(int) * params->nx * params->ny,64);
  
    for(int jj=0;jj<params->ny;jj++){
        for( int ii=0;ii<params->nx;ii++){
            full_grid[jj*params->nx+ii] = 0;
        }
    }


  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }
  obs = params->nx * params->ny;

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);
    
    // update counter before setting values !!
    if(full_grid[yy*params->nx+xx]==0) --obs;
    
    full_grid[yy*params->nx+xx]=blocked;
    
  }
  
  fclose(fp);
  
}
  MPI_Datatype obstacles_row;
  MPI_Type_contiguous(params->nx,MPI_INT,&obstacles_row);
  MPI_Type_commit(&obstacles_row);
  MPI_Bcast(&obs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Scatterv(full_grid, nb_rows_per_rank, offset_index, obstacles_row,
              &(*obstacles_ptr)[params->nx], nb_rows_per_rank[rank], obstacles_row,
               0, MPI_COMM_WORLD);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);
  *av_vels_per_rank = (float*)malloc(sizeof(float) * params->maxIters);

  return obs;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;
  

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles,int rows,int cols,int offset,int denom)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
  float tmp = av_velocity(params, cells, obstacles,rows,cols,offset,denom) * params.reynolds_dim;
  float tmp2 = viscosity  ; 

  return  tmp/(float) tmp2;
}

float total_density(const t_param params, t_speed* cells,int rows,int cols)
{
  float total = 0.f;  /* accumulator */
	float output = 0.f;

  for (int jj = 0; jj < rows ; jj++)
  {
    for (int ii = 0; ii < cols; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + (jj+1)*cols].speeds[kk];
      }
    }
  }
  MPI_Reduce(&total, &output, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  total = output;
  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels, char* mode , int rows,int offset)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, mode);

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 1; jj < rows+1 ; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }
      int offset_write = jj + offset - 1;

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, offset_write, u_x, u_y, u, pressure, obstacles[jj * params.nx + ii]);
    }
  }

  fclose(fp);

  return EXIT_SUCCESS;
}
void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
