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
    set_title("Gtkmm Programming - C++");
    set_position(Gtk::WIN_POS_CENTER);

    box.add(img);
    add(box);

    box.set_events(Gdk::BUTTON_PRESS_MASK);
    box.signal_button_press_event().connect(
            sigc::mem_fun(*this, &Simulation::on_eventbox_button_press) );



    int middle_x = width / 2;
    int middle_y = height / 2;

    img_data = new guint8[3*height*width];

    data_array_new = new float*[height];
    for(int i = 0; i < height; ++i) data_array_new[i] = new float[width];

    data_array_old = new float*[height];
    for(int i = 0; i < height; ++i) data_array_old[i] = new float[width];

    // populate data array (anfangsbedingungen)
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            data_array_new[i][j] = 0.0;
            if (i > 1.6*middle_x and j > 1.6*middle_y)
            {
                data_array_new[i][j] = 1.0f;
            }
            if ((i-middle_x)*(i-middle_x) + (j-middle_y)*(j-middle_y) < 100*100)
            {
                data_array_new[i][j] = 0.8;
            }
        }
    }

    u = new float[size];
    v = new float[size];
    u_prev = new float[size];
    v_prev = new float[size];
    dens = new float[size];
    dens_prev = new float[size];
    occupiedGrid = new bool[size];
    Nx = new float[size];
    Ny = new float[size];

    initializeGrid();
    initializeFluid(u,v,dens);


    // create slot for timeout signal
    int timeout_value = 50; //in ms
//    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout);
    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout_2);

    //connect slot to signal
    Glib::signal_timeout().connect(my_slot, timeout_value);

    show_all_children();


    update_view_2(dens);
//    update_view(data_array_new);


}

Simulation::~Simulation(){
}




bool Simulation::on_timeout(){

    time_step_counter += 1;

    // update old_data
    data_array_old = data_array_new;

    float clf = 0.42; // must be smaller than 0.5, otherwise instable!

    std::clock_t start;
    start = std::clock();

    // update data array
    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            float lower = data_array_new[i][j-1];
            float upper = data_array_new[i][j+1];
            float left = data_array_new[i-1][j];
            float right = data_array_new[i+1][j];
            // diffuse
            data_array_new[i][j] = data_array_old[i][j] + clf*(lower + upper + left + right - 4*data_array_old[i][j]);
        }
    }

    // your test
    std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    // update view of data array
    update_view(data_array_new);

    return true;
}

bool Simulation::on_timeout_2() {

    time_step_counter += 1;

    cout<< "Iteration " << time_step_counter << endl;


    std::clock_t start;
    start = std::clock();

    get_from_UI( dens,u,v,dens_prev, u_prev, v_prev ,t); // external influence on fluid

    // NAVIER-STOKES SOLUTION: VELOCITY FIELD AND DENSITY FIELD SEPARATELY SOLVED
    vel_step( u, v, u_prev, v_prev, visc, dt);
    dens_step( dens, dens_prev, u, v, diff, dt);

    std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    update_view_2(dens);

}



void Simulation::update_view(float ** data_array_new)
{


    // update view of data array
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int ii = 3*(i * width + j);

            double value = data_array_new[i][j] * 255.0;

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



void Simulation::update_view_2(float * dens)
{

//    printf("Density Array: \n");
//
//    for (int i = 0; i < width; i+=1)
//    {
//        for (int j = 0; j < height; j+=1)
//        {
//            printf("%f ", dens[IX(i,j)]);
//        }
//        printf("\n");
//    }

    // update view of data array
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            int ii = 3*(i * width + j);

            float value = dens[IX(i,j)] * 255.0;
//            float value = float(i) / width * 255;
//            printf("%f ", value);

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

    printf("x = %f, y = %f\n", x, y);

    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {
            if ((i - y)*(i - y) + (j - x)*(j - x) < 10*10)
            {
//                data_array_old[i][j] = 1.0f;
                printf("adding some density\n");
                dens_prev[IX(i,j)] = 1.0;
                dens[IX(i,j)] = 1.0;
            }
        }
    }

    return true;
}


void Simulation::initializeGrid()
// initialize the grid cell coordinates
{
    for ( int i=1 ; i<=N ; i++ ) {
        Nx[i] = i;
        Ny[i] = i;
        occupiedGrid[i] = false;
    }

}


void Simulation::initializeFluid(float *u,float *v,float *dens)
{
    // initialize the fluid state (velocities and density) at t=0
    for ( int i=1 ; i<=N ; i++ ) {
        for ( int j=1 ; j<=N ; j++ ) {
            u[IX(i,j)] = 1.0; // u velocity at t=0
            v[IX(i,j)] = 0.00; // v velocity at t=0

            dens[IX(i,j)] = 0.0; // density at t=0
            if (i < 100 and i > 50) dens[IX(i,j)] = 1.;
        }
    }
}

void Simulation::diffuse(int b, float * x, float * x0, float diff, float dt )
{
    // diffusion step is obtained by Gauss-Seidel relaxation equation system solver
    // used for density, u-component and v-component of velocity field separately

    float a=dt*diff*N*N;
    int nIter = 20;

    for (int k=0 ; k < nIter ; k++ ) {
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

    int i, j, i0, j0, i1, j1;
    float x, y, s0, t0, s1, t1, dt0;
    bool occ;
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

            occ = occupiedGrid[IX(i,j)];

            if(occ == 0) d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
            else
                d[IX(i,j)] = 0;
        }
    }

    set_bnd(b, d );
}



void Simulation::add_source (float * x, float * s, float dt )
{
    // add sources for velocity field or density field
    int i, size=(N+2)*(N+2);
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

    for (int i=1 ; i<=N; i++ ) {

        // left border

        x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)]; // bounded box boundary conditions
        x[IX(0, i)] = 0;

        // right border
        x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)]; // bounded box boundary conditions
        x[IX(N + 1, i)] = 0;


        // bottom border
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)]; // bounded box boundary conditions
        x[IX(i, 0)] = 0;

//        // make jets at the bottom
//        x[IX(i,0 )] = 0;
//
//        if ((i > 4.5*N/10 && i < 5.5*N/10))
//            x[IX(i,0 )] += 0.4;
//        else if ((i > 0.5*N/10 && i < 1.8*N/10))
//            x[IX(i,0 )] += 0.4;
//        else if ((i > 7.8*N/10 && i < 8.9*N/10))
//            x[IX(i,0 )] += 0.4;


        // upper border
        x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x[IX(i, N)]; // bounded box boundary conditions
        x[IX(i, N + 1)] = 0;
    }



    // implementing internal flow obstacles
    if(b != 0) {  // only changed boundaries for flow -> b = 1,2
        for ( int i=1 ; i<=N ; i++ ) {
            for ( int j=1 ; j<=N ; j++ ) {
                    x[IX(i-1,j)] = b==1 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i+1,j)] = b==1 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i,j-1 )] = b==2 ? -x[IX(i,j)] : x[IX(i,j)];
                    x[IX(i,j+1)] = b==2 ? -x[IX(i,j)] : x[IX(i,j)];

            }
        }
    }

    // define edge cells as median of neighborhood
    x[IX(0 ,0 )] = 0.5*(x[IX(1,0 )]+x[IX(0 ,1)]);
    x[IX(0 ,N+1)] = 0.5*(x[IX(1,N+1)]+x[IX(0 ,N )]);
    x[IX(N+1,0 )] = 0.5*(x[IX(N,0 )]+x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5*(x[IX(N,N+1)]+x[IX(N+1,N )]);
}



void Simulation::get_from_UI(float *dens, float *u, float *v, float *dens_prev, float *u_prev, float *v_prev, float t)
{
    // adds density and/or velocity in this function
    for ( int i=1 ; i<=N ; i++ ) {
        for ( int j=1 ; j<=N ; j++ ) {

            // Density input for nice jets
            if ((i > 4.6*N/10 && i < 5.4*N/10) && (j<10))
                dens[IX(i,j)] = 1.0;
        }
    }
}




