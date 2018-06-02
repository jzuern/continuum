//
// Created by jannik on 5/17/18.
//

#ifndef PROJ_SIMULATION_H
#define PROJ_SIMULATION_H

#include <stdexcept>
#include "numerical_kernels.h"


#define USE_CUDA 1
#define IX(i,j) ((i)+(width+2)*(j))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}

using namespace std;



class Simulation : public Gtk::Window{


public:
    // i -> Index right/left (x)
    // j -> Index up/down (y)
    // u -> velocity right/left
    // v -> velocity up/down
    // width: x size
    // height: y size

    // GUI members
    Gtk::Image img;
    Gtk::EventBox box;
    guint8 * img_data;

    // simulation stuff
    int time_step_counter = 0;
    const float dt = 0.0001; // incremental time step length
    const int height = 350;
    const int width = 350;
    const int size = (height+2)*(width+2); // grid size incl. boundaries

    // numerical parameters
    int maxiter = 10; // higher -> more accurate


    void initializeGrid(); // initialize grid variables
    void initializeFluid(); // initialize fluid velocities and density distribution at t=0

    // fluid dynamics routines

    void vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt); // determine velocity vectors in next time step
    void dens_step(float *& dens, float *dens_prev, float *u, float *v, float diff, float dt); // determine fluid field in next time step

    void project(float *u, float *v, float *u0, float *v0 );
    void project_gpu(float *u, float *v, float *u0, float *v0 );


    void add_source(float *x, float *x0, float dt );
    void add_source_gpu(float *x, float *x0, float dt );
    void set_bnd(int b, float *x);
    void set_bnd2();

    void diffuse(int,float *x,float *x0, float diff, float dt);
    void diffuse_gpu(int, float * x,float * x0, float diff, float dt);

    void advect(int b, float * d, float * d0, float * u, float * v, float dt );
    void advect_gpu(int b, float * d, float * d0, float * u, float * v, float dt, bool * occ);

    // cuda code

    // GUI handling functions
    bool on_timeout_cfd(); //return true to keep the timeout and false to end it
    bool on_timeout_heat(); //return true to keep the timeout and false to end it

    void update_view(float * dens);
    bool get_mouse_event(GdkEventButton*);

    // constructor
    Simulation();

    // destructor
    virtual ~Simulation();


    // fluid field variables
    float * u;
    float * v;
    float * u_prev;
    float * v_prev;
    float * dens;
    float * dens_prev;
    bool * occupiedGrid; // define flow obstacles

    float visc = 0.001; // viscosity
    float diff = 0.01; // diffusion rate

private:


};



#endif //PROJ_SIMULATION_H
