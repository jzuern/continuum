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
//    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout_cfd);
    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout_heat);

    //connect slot to signal
    Glib::signal_timeout().connect(my_slot, timeout_value);

    show_all_children();
    update_view(dens);
    printf("completed update_view\n");

}

Simulation::~Simulation()
{
}

bool Simulation::on_timeout_heat() {

    cout<< "Iteration " << time_step_counter << endl;



    std::clock_t start;
    start = std::clock();

    // solve heat equation
    SWAP(dens, dens_prev);


//    printf("on_timeout_heat dens before heat diffusion: \n");
//    pretty_printer(dens,height, width);

    float * h_T_GPU_result = (float *)malloc((width+2) * (height+2) * sizeof(float));

    try_diffuse(dens,dens_prev,h_T_GPU_result, height,width);
    dens = h_T_GPU_result;


    float time = (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000); // ms

    std::cout << "    Step calculation duration: " << time << " ms" << std::endl;
    update_view(dens);
    printf("completed update_view\n");
    time_step_counter += 1;


}

bool Simulation::on_timeout_cfd() {

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

            float value = dens[IX(i,j)] * 255.0;

            img_data[ii] = guint8(value);
            img_data[ii+1] = guint8(value);
            img_data[ii+2] = guint8(value);
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
                dens[IX(i,j)] = 1.0;
                dens_prev[IX(i,j)] = 1.0;

                //u[IX(i,j)] += 0.4;
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

            if (i > width/2.0) dens[IX(i,j)] = 0.8;
            if (i > width/2.0) dens_prev[IX(i,j)] = 0.8;

        }
    }
}

void Simulation::diffuse(int b, float * x, float * x0, float diff, float dt )
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately

    float a=dt*diff*height*width;

    for (int k=0 ; k < gauss_seidel_iterations ; k++ )
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

            d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+
                             t1*d0[IX(i0,j1)])+
                         s1*(t0*d0[IX(i1,j0)]+
                             t1*d0[IX(i1,j1)]);
        }
    }

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


void Simulation::dens_step (float * x, float * x0, float * u, float * v, float diff,float dt)
{

    // executes all routines for motion of density field in one time step
    add_source(x, x0, dt );
    SWAP ( x0,x);
    diffuse(0, x, x0, diff, dt );
    SWAP ( x0,x);
    advect(0, x, x0, u, v, dt );
    set_bnd2();
}


void Simulation::vel_step (float * u, float * v, float *  u0, float * v0,float visc, float dt )
{
    set_bnd2();

    // executes all routines for motion of velocity field in one time step
    add_source ( u, u0, dt );
    SWAP ( u0, u );
    diffuse(1, u, u0, visc, dt);

    add_source ( v, v0, dt );
    SWAP ( v0, v );
    diffuse(2, v, v0, visc, dt);

    project ( u, v, u0, v0 );

    SWAP ( u0, u );
    SWAP ( v0, v );

    advect(1, u, u0, u0, v0, dt );
    advect(2, v, v0, u0, v0, dt );

    project (u, v, u0, v0 );

    set_bnd2();

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

    for (int k=0 ; k<gauss_seidel_iterations ; k++ )
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
    set_bnd2();
}

void Simulation::set_bnd(int b, float * x)
{
    // define boundary values for velocity and density
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

        if ((i > 0.46*height && i < 0.5*height)) x[IX(i,1)] = 0.5;

    }

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

    // define edge cells as median of neighborhood
    x[IX(0 ,0 )] = 0.5f*(x[IX(1,0 )]+x[IX(0 ,1)]);
    x[IX(0 ,height+1)] = 0.5f*(x[IX(1,height+1)]+x[IX(0 ,height)]);
    x[IX(width+1,0 )] = 0.5f*(x[IX(width,0 )]+x[IX(width+1,1)]);
    x[IX(width+1,height+1)] = 0.5f*(x[IX(width,height+1)]+x[IX(width+1,height)]);
}


void Simulation::set_bnd2()
{
    // define boundary values for velocity and density
    for (int i=0 ; i<height+2; i++ ) {

        for (int j = 0; j<width/5; j++)
        {
            if ((i > 0.1*height && i < 0.15*height))
            {
                u[IX(i,j)] = -0.5;
                dens[IX(i,j)] = 0.5;
                v[IX(i,j)] = 0.5;

                u_prev[IX(i,j)] = 0.5;
                dens_prev[IX(i,j)] = 0.5;
                v_prev[IX(i,j)] = 0.5;
            }
        }
    }
}





