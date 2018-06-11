//
// Created by jannik on 5/17/18.

#include <iostream>
#include <gtkmm.h>
#include "simulation.h"
#include "numerical_kernels.h"
#include <cmath>        // std::abs
#include <ctime>  // clock


using namespace std;




Simulation::Simulation(){

    set_default_size(width,height);
    set_title("Continuum");
    set_position(Gtk::WIN_POS_CENTER);

    box.add(img);
    add(box);

    box.set_events(Gdk::BUTTON_PRESS_MASK);
    box.signal_button_press_event().connect(
            sigc::mem_fun(*this, &Simulation::get_mouse_event) );

    img_data = new guint8[3*size];

    u = new float[size];
    v = new float[size];
    u_prev = new float[size];
    v_prev = new float[size];
    dens = new float[size];
    dens_prev = new float[size];
    occupiedGrid = new bool[size];

    // grid initialization
    initializeGrid();
    initializeFluid();

    // create slot for timeout signal
    int timeout_value = 50; //in ms
    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout);

    //connect slot to signal
    Glib::signal_timeout().connect(my_slot, timeout_value);

    show_all_children();
    update_view(dens);
    printf("completed update_view\n");

}

Simulation::~Simulation()
{
}


bool Simulation::on_timeout() {

    cout<< "Iteration " << time_step_counter << endl;

    std::clock_t start;
    start = std::clock();

    // NAVIER-STOKES SOLUTION: VELOCITY FIELD AND DENSITY FIELD SEPARATELY SOLVED
    vel_step( u, v, u_prev, v_prev, visc, dt);
    dens_step( dens, dens_prev, u, v, diff, dt);
    time_step_counter += 1;

    float time = (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000); // ms

    std::cout << "    Step calculation duration: " << time << " ms" << std::endl;
    update_view(dens);
    printf("completed update_view\n");

}


void Simulation::update_view(float * dens)
{
    // update view of data array
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int ii = 3*(i * width + j);

            if (occupiedGrid[IX(i,j)])
            {
                img_data[ii] = guint8(255);
                img_data[ii+1] = guint8(255);
                img_data[ii+2] = guint8(255);
            }
            else
            {
//                float value_u = u[IX(i,j)];
//                float value_v = v[IX(i,j)];
                float value_dens = dens[IX(i,j)] * 255.0 / 5.0;

                img_data[ii] = guint8(value_dens);
                img_data[ii+1] = guint8(value_dens);
                img_data[ii+2] = guint8(value_dens);
            }

        }
    }

    int rowstride = 3*width;
    bool has_alpha = false;
    int bits_per_sample = 8;


    Glib::RefPtr<Gdk::Pixbuf> ref_dest =
            Gdk::Pixbuf::create_from_data (
                    img_data, Gdk::COLORSPACE_RGB,has_alpha,bits_per_sample,width,height,rowstride);


    img.clear();
    img.set(ref_dest);
}


bool Simulation::get_mouse_event(GdkEventButton* e)
{
    gdouble x = e->x;
    gdouble y = e->y;

    float radius = 10*10;

    printf("found mouse click at x = %f, y = %f\n", x, y);

    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            if ((i - y)*(i - y) + (j - x)*(j - x) < radius)
            {
//                dens[IX(i,j)] = 1.0;
//                dens_prev[IX(i,j)] = 1.0;
                occupiedGrid[IX(i,j)] = true;
            }
        }
    }

    return true;
}


void Simulation::initializeGrid()
// initialize the grid cell coordinates
{
    // initialize occ grid
    for ( int i=1 ; i<=width ; i++ )
    {
        for ( int j=1 ; j<=height ; j++ )
        {
            occupiedGrid[IX(i,j)] = false; // u velocity at t=0
        }
    }
}


void Simulation::initializeFluid()
{
    // initialize the fluid state (velocities and density) at t=0
    for ( int i=0 ; i<=width+1 ; i++ )
    {
        for ( int j=0 ; j<=height+1 ; j++ )
        {
            u[IX(i,j)] = 0.0; // u velocity at t=0
            v[IX(i,j)] = 0.0; // v velocity at t=0
            dens[IX(i,j)] = 0.0; // density at t=0

            u_prev[IX(i,j)] = 0.0; // u velocity at t=0
            v_prev[IX(i,j)] = 0.0; // v velocity at t=0
            dens_prev[IX(i,j)] = 0.0; // density at t=0
        }
    }
}

void Simulation::diffuse(int b, float * x, float * x0, float diff, float dt )
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately

    float a=dt*diff*height*width;

    for (int k=0 ; k < maxiter ; k++ )
    {
        for (int i=1 ; i<=width ; i++ )
        {
            for (int j=1 ; j<=height ; j++ ) {
                x[IX(i, j)] = (x0[IX(i, j)] +
                               a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                              (1 + 4 * a);
            }
        }
    }
    set_bnd(b, x );
}

void Simulation::diffuse_gpu(int b, float * x, float * x_old, float diff, float dt)
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately

    try_diffuse(x, x_old, height, width, dt, diff, maxiter);

    set_bnd(b, x );
}




void Simulation::advect(int b, float * d, float * d0, float * u, float * v, float dt )
{
    // calculate the advection of density in velocity field and velocity field along itself

    // b == 0: density
    // b == 1: u
    // b == 2: v

    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    dt0 = dt*max(width,height);
    for ( i=1 ; i<=width ; i++ )
    {
        for ( j=1 ; j<=height ; j++ )
        {

            x = i-dt0*u[IX(i,j)];
            y = j-dt0*v[IX(i,j)];

            if (x<0.5) x=0.5;
            if (x>width+0.5) x=width+ 0.5; i0=(int)x; i1=i0+ 1;
            if (y<0.5) y=0.5;
            if (y>height+0.5) y=height+ 0.5; j0=(int)y; j1=j0+1;
            s1 = x-i0;
            s0 = 1-s1;
            t1 = y-j0;
            t0 = 1-t1;

            bool occ = occupiedGrid[IX(i,j)];

            if(occ == 0) d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
            else
                d[IX(i,j)] = 0;
        }
    }

    set_bnd(b, d );
}

void Simulation::advect_gpu(int b, float * d, float * d0, float * u, float * v, float dt, bool * occ )
{
    try_advect( d,  d0,  u,  v, height, width, dt, occ);
    set_bnd(b, d );
}



void Simulation::add_source (float * x, float * s, float dt )
{
    // add sources for velocity field or density field
    for ( int i=0 ; i<size ; i++ )
    {
        x[i] += dt*s[i];
    }
}

void Simulation::add_source_gpu (float * x, float * s, float dt )
{
    try_source(x, s, height, width, dt);
}


void Simulation::dens_step (float *& x, float * x0, float * u, float * v, float diff,float dt)
{

    // executes all routines for motion of density field in one time step

    // allocate cuda memory
    const int NX = width+2;			// --- Number of discretization points along the x axis
    const int NY = height+2;			// --- Number of discretization points along the y axis

    // allocate cuda memory
    float *d_x;			gpuErrchk(cudaMalloc((void**)&d_x,			NX * NY * sizeof(float)));
    float *d_x0;			gpuErrchk(cudaMalloc((void**)&d_x0,			NX * NY * sizeof(float)));

    // copy host memory to device memory
    gpuErrchk(cudaMemcpy(d_x, x,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_x0, x0,	 NX * NY * sizeof(float), cudaMemcpyHostToDevice));

    // --- Grid size
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid (iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));

    call_add_source_kernel(d_x,d_x0, NX, NY,dt)

    add_source_kernel<<<dimGrid, dimBlock>>>(d_x,d_x0, NX, NY,dt);

    // --- Copy results from device to host
    gpuErrchk(cudaMemcpy(x,	 d_x,	  NX * NY * sizeof(float), cudaMemcpyDeviceToHost));

    // free device memory
    gpuErrchk(cudaFree(d_x));
    gpuErrchk(cudaFree(d_x0));




#if USE_CUDA
    //add_source_gpu(x, x0, dt );
    SWAP ( x0,x);
    diffuse_gpu(0, x0,x, diff, dt );
    SWAP ( x0,x);
    advect_gpu(0, x, x0, u, v, dt , occupiedGrid);
#else
    add_source(x, x0, dt );
    SWAP ( x0,x);
    diffuse(0, x, x0, diff, dt );
    SWAP ( x0,x);
    advect(0, x, x0, u, v, dt );
#endif


}


void Simulation::vel_step (float * u, float * v, float *  u0, float * v0,float visc, float dt )
{
    // executes all routines for motion of velocity field in one time step

    // GPU
#if USE_CUDA
    add_source_gpu ( u, u0, dt );
    SWAP ( u0, u );
    diffuse_gpu(1, u0,u, diff, dt );
    add_source_gpu( v, v0, dt );
    SWAP ( v0, v );
    diffuse_gpu(2, v0, v, diff, dt );
    project_gpu ( u, v, u0, v0, dens);
    SWAP ( u0, u );
    SWAP ( v0, v );
    advect_gpu(1, u, u0, u0, v0, dt,occupiedGrid );
    advect_gpu(2, v, v0, u0, v0, dt,occupiedGrid);
    project_gpu ( u, v, u0, v0 , dens);
#else
    add_source ( u, u0, dt );
    SWAP ( u0, u );
    diffuse(1, u0,u, diff, dt );
    add_source( v, v0, dt );
    SWAP ( v0, v );
    diffuse(2, v0, v, diff, dt );
    project ( u, v, u0, v0);
    SWAP ( u0, u );
    SWAP ( v0, v );
    advect(1, u, u0, u0, v0, dt );
    advect(2, v, v0, u0, v0, dt);
    project ( u, v, u0, v0);
#endif

}

void Simulation::project (float * u, float * v, float * p, float * div )
{
    // force routing to be mass conserving (use "hodge decomposition" for obtained velocity field and
    // eliminate gradient field)
    // this will make the velocity field to have fluid-like swirls as desired

    float h;
    h = 1.0/max(height,width);

    for (int i=1 ; i<=width ; i++ )
    {
        for (int j=1 ; j<=height ; j++ )
        {
            div[IX(i,j)] = -0.5*h*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)]);
            p[IX(i,j)] = 0;
        }
    }

    set_bnd(0, div );
    set_bnd(0, p );

    for (int k=0 ; k<maxiter ; k++ )
    {
        for (int i=1 ; i<=width ; i++ )
        {
            for (int j=1 ; j<=height ; j++ )
            {
                p[IX(i,j)] = (div[IX( i,j)]+p[IX(i-1,j)]+p[IX(i+1,j)]+p[IX(i,j-1)]+p[IX(i,j+1)])/4;
            }
        }
        set_bnd(0, p );
    }

    for (int i=1 ; i<=width ; i++ )
    {
        for (int j=1 ; j<=height ; j++ )
        {
            u[IX(i,j)] -= 0.5*(p[IX(i+1,j)]-p[IX(i-1,j)])/h;
            v[IX(i,j)] -= 0.5*(p[IX(i,j+1)]-p[IX(i,j-1)])/h;
        }
    }
    set_bnd( 1, u );
    set_bnd( 2, v );
}


void Simulation::project_gpu(float * u, float * v, float * p, float * div, float * dens)
{
    // force routing to be mass conserving (use "hodge decomposition" for obtained velocity field and
    // eliminate gradient field)
    // this will make the velocity field to have fluid-like swirls as desired

    float h = 1.0/max(height,width);

    try_project_1(div, u,v,p, height, width, h);

    set_bnd(0, div );
    set_bnd(0, p );



//    try_project_2(p,div, height, width, maxiter, occupiedGrid, dens, u);

    for (int k=0 ; k<maxiter ; k++ )
    {
        for (int i=1 ; i<=width ; i++ )
        {
            for (int j=1 ; j<=height ; j++ )
            {
                p[IX(i,j)] = (div[IX( i,j)]+p[IX(i-1,j)]+p[IX(i+1,j)]+p[IX(i,j-1)]+p[IX(i,j+1)])/4;
            }
        }
        set_bnd(0, p );
    }

    try_project_3(u,v,p,height,width,h);

    set_bnd( 1, u );
    set_bnd( 2, v );
}

void Simulation::set_bnd(int b, float * x)
{
    // define boundary values for velocity and density

    // left and right wall
    for (int i=0 ; i<height+2; i++ ) {

        if (b == 0) // density
        {
            x[IX(0,i)] = x[IX(1,i)]; // left
            x[IX(width+1,i)] = x[IX(width,i)];// right
        }

        if (b == 1) // u velocity component
        {
            x[IX(0,i)] = -x[IX(1,i)];// left
            x[IX(width+1,i)] = -x[IX(width,i)];// right
        }

        if (b == 2) // v velocity component
        {
            x[IX(0,i)] = x[IX(1,i)]; // left
            x[IX(width+1,i)] = x[IX(width,i)]; // right
        }

        // additional boundary conditions:

        if ((i > 0.4*height && i < 0.5*height)){
            dens[IX(1,i)] = 1.0;
            u[IX(1,i)] = 10.0;
            //v[IX(1,i)] = -5.0;

            dens[IX(i,1)] = 0.6;
            u[IX(i,1)] = 10.0;

        }
    }

    // upper and lower wall

    for (int i=0 ; i<width+2; i++ ) {

        if (b == 0) // density
        {
            x[IX(i,0 )] = x[IX(i,1)];// bottom
            x[IX(i,height+1)] = x[IX(i,height)]; // top
        }

        if (b == 1) // u velocity component
        {
            x[IX(i,0 )] = x[IX(i,1)];// bottom
            x[IX(i,height+1)] = x[IX(i,height)];// top
        }

        if (b == 2) // v velocity component
        {
            x[IX(i,0 )] = -x[IX(i,1)]; // bottom
            x[IX(i,height+1)] = -x[IX(i,height)];// top
        }
    }

    // implementing internal flow obstacles
    if(b != 0) {  // only changed boundaries for flow -> b = 1,2
        for ( int i=1 ; i<=height ; i++ ) {
            for ( int j=1 ; j<=width ; j++ ) {
                bool occ = occupiedGrid[IX(i,j)];
                if(occ == 1){
                    x[IX(i-1,j)] = b==1 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i+1,j)] = b==1 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i,j-1 )] = b==2 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i,j+1)] = b==2 ? -x[IX(i,j)] : x[IX(i,j)];
                }
            }
        }
    }

    // define edge cells as median of neighborhood
    x[IX(0 ,0 )] = 0.5f*(x[IX(1,0 )]+x[IX(0 ,1)]);
    x[IX(0 ,height+1)] = 0.5f*(x[IX(1,height+1)]+x[IX(0 ,height)]);
    x[IX(width+1,0 )] = 0.5f*(x[IX(width,0 )]+x[IX(width+1,1)]);
    x[IX(width+1,height+1)] = 0.5f*(x[IX(width,height+1)]+x[IX(width+1,height)]);
}





