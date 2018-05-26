//
// Created by jannik on 5/17/18.

#include <iostream>
#include <gtkmm.h>
#include "simulation.h"
#include <cmath>        // std::abs
#include <ctime>  // clock
#include "numerical_kernels.h"

using namespace std;


Simulation::Simulation(){

    set_default_size(width, height);
    set_title("Continuum");
    set_position(Gtk::WIN_POS_CENTER);

    box.add(img);
    add(box);

    box.set_events(Gdk::BUTTON_PRESS_MASK);
    box.signal_button_press_event().connect(
            sigc::mem_fun(*this, &Simulation::on_eventbox_button_press) );


    int middle_x = width / 2;
    int middle_y = height / 2;

    img_data = new guint8[3*height*width];

    u = new float[size];
    v = new float[size];
    u_prev = new float[size];
    v_prev = new float[size];
    dens = new float[size];
    dens_prev = new float[size];
    occupiedGrid = new bool[size];

    initializeGrid();
    initializeFluid();

    // create slot for timeout signal
    int timeout_value = 50; //in ms
    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout);

    //connect slot to signal
    Glib::signal_timeout().connect(my_slot, timeout_value);

    show_all_children();

    update_view(dens);

}

Simulation::~Simulation(){
}

void Simulation::print_helper()
{
    int every_n = 1;

    printf("Velocity u Array: \n");

    for (int i = 0; i < width; i+=every_n)
    {
        for (int j = 0; j < height; j+=every_n)
        {
            printf("%.5e ", u[IX(i,j)]);
        }
        printf("\n");
    }

    printf("Velocity u_prev Array: \n");

    for (int i = 0; i < width; i+=every_n)
    {
        for (int j = 0; j < height; j+=every_n)
        {
            printf("%.5e ", u_prev[IX(i,j)]);
        }
        printf("\n");
    }
}


bool Simulation::on_timeout() {

    time_step_counter += 1;
    cout<< "Iteration " << time_step_counter << endl;
    std::clock_t start;
    start = std::clock();

    // NAVIER-STOKES SOLUTION: VELOCITY FIELD AND DENSITY FIELD SEPARATELY SOLVED
    vel_step( u, v, u_prev, v_prev, visc, dt);
    printf("completed vel_step ");

    dens_step( dens, dens_prev, u, v, diff, dt);
    printf("completed dens_step \n");

    std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
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
//            float value = float(i) / width * 255;

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


bool Simulation::on_eventbox_button_press(GdkEventButton* e)
{
    gdouble x = e->x;
    gdouble y = e->y;

    printf("found mouse click at x = %f, y = %f\n", x, y);

    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            if ((i - y)*(i - y) + (j - x)*(j - x) < 5*5)
            {
                dens[IX(i,j)] = 1.0;
                u[IX(i,j)] += 0.4;
            }
        }
    }

    return true;
}


void Simulation::initializeGrid()
// initialize the grid cell coordinates
{
    // initialize occ grid
    for ( int i=1 ; i<=N ; i++ ) {
        for ( int j=1 ; j<=N ; j++ ) {
            occupiedGrid[IX(i,j)] =false; // u velocity at t=0
        }
    }
}


void Simulation::initializeFluid()
{
    // initialize the fluid state (velocities and density) at t=0
    for ( int i=0 ; i<=N ; i++ ) {
        for ( int j=0 ; j<=N ; j++ ) {
            u[IX(i,j)] = 0.1; // u velocity at t=0
            v[IX(i,j)] = 0.0; // v velocity at t=0
            dens[IX(i,j)] = 1.0; // density at t=0
        }
    }
}

void Simulation::diffuse(int b, float * x, float * x0, float diff, float dt )
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately

    float a=dt*diff*N*N;

    for (int k=0 ; k < gauss_seidel_iterations ; k++ ) {
        for (int i=1 ; i<=N ; i++ ) {
            for (int j=1 ; j<=N ; j++ ) {
                x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/(1+4*a);
            }
        }
        set_bnd(b, x );
    }
}


void Simulation::advect(int b, float * d, float * d0, float * u, float * v, float dt )
{
    // calculate the advection of density in velocity field and velocity field along itself

    // b == 0: density
    // b == 1: u
    // b == 2: v

    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    dt0 = dt*N;

    for ( i=1 ; i<=N ; i++ ) {
        for ( j=1 ; j<=N ; j++ ) {

            x = i-dt0*u[IX(i,j)];
            y = j-dt0*v[IX(i,j)];

            if (x<0.5) x=0.5;
            if (x>N+0.5) x=N+ 0.5; i0=(int)x; i1=i0+ 1;
            if (y<0.5) y=0.5;
            if (y>N+0.5) y=N+ 0.5; j0=(int)y; j1=j0+1;
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
    int i, size=(height+2)*(width+2);
    for ( i=0 ; i<size ; i++ ){
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
}


void Simulation::vel_step (float * u, float * v, float *  u0, float * v0,float visc, float dt )
{
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
}

void Simulation::project (float * u, float * v, float * p, float * div )
{
    // force routing to be mass conserving (use "hodge decomposition" for obtained velocity field and
    // eliminate gradient field)
    // this will make the velocity field to have fluid-like swirls as desired

    float h;
    h = 1.0/N;

    for (int i=1 ; i<=N ; i++ ) {
        for (int j=1 ; j<=N ; j++ ) {
            div[IX(i,j)] = -0.5*h*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)]);
            p[IX(i,j)] = 0;
        }
    }

    set_bnd(0, div );
    set_bnd(0, p );

    for (int k=0 ; k<20 ; k++ ) {
        for (int i=1 ; i<=N ; i++ ) {
            for (int j=1 ; j<=N ; j++ ) {
                p[IX(i,j)] = (div[IX( i,j)]+p[IX(i-1,j)]+p[IX(i+1,j)]+p[IX(i,j-1)]+p[IX(i,j+1)])/4;
            }
        }
        set_bnd(0, p );
    }

    for (int i=1 ; i<=N ; i++ ) {
        for (int j=1 ; j<=N ; j++ ) {
            u[IX(i,j)] -= 0.5*(p[IX(i+1,j)]-p[IX(i-1,j)])/h;
            v[IX(i,j)] -= 0.5*(p[IX(i,j+1)]-p[IX(i,j-1)])/h;
        }
    }
    set_bnd( 1, u );
    set_bnd( 2, v );
}

void Simulation::set_bnd(int b, float * x)
{
    // define boundary values for velocity and density


    for (int i=0 ; i<=N+1; i++ ) {

        if (b == 0) // density
        {
            x[IX(0,i)] = x[IX(1,i)]; // left
            x[IX(N+1,i)] = x[IX(N,i)];// right
            x[IX(i,0 )] = x[IX(i,1)];// bottom
            x[IX(i,N+1)] = x[IX(i,N)]; // top
        }

        if (b == 1) // u velocity component
        {
            x[IX(0,i)] = -x[IX(1,i)];// left
            x[IX(N+1,i)] = -x[IX(N,i)];// right
            x[IX(i,0 )] = x[IX(i,1)];// bottom
            x[IX(i,N+1)] = x[IX(i,N)];// top
        }

        if (b == 2) // v velocity component
        {
            x[IX(0,i)] = x[IX(1,i)]; // left
            x[IX(N+1,i)] = x[IX(N,i)]; // right
            x[IX(i,0 )] = -x[IX(i,1)]; // bottom
            x[IX(i,N+1)] = -x[IX(i,N)];// top
        }


//        if ((i > 0.0*N && i < 0.5*N))
//            x[IX(i,1)] = 0.8;

        if (b == 1 and i > 0.5*N) // u velocity component
        {
            x[IX(1,i)] = 1.0;// left
        }

//        if ((i > 0.5*N && i < 1.0*N))
//            x[IX(i,N)] = -0.8;

    }

    // define edge cells as median of neighborhood
    x[IX(0 ,0 )] = 0.5*(x[IX(1,0 )]+x[IX(0 ,1)]);
    x[IX(0 ,N+1)] = 0.5*(x[IX(1,N+1)]+x[IX(0 ,N )]);
    x[IX(N+1,0 )] = 0.5*(x[IX(N,0 )]+x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5*(x[IX(N,N+1)]+x[IX(N+1,N )]);
}





