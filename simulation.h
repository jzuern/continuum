//
// Created by jannik on 5/17/18.
//

#ifndef PROJ_SIMULATION_H
#define PROJ_SIMULATION_H

#include <stdexcept>

#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}



using namespace std;

class Simulation : public Gtk::Window{
public:

    // GUI stuff
    Gtk::Image img;
    Gtk::EventBox box;
    guint8 * img_data;

    // simulation stuff
    int time_step_counter = 0;

    // numerical parameters

    int gauss_seidel_iterations = 20;

    const int height = 250;
    const int width = 250;
    const int N = height;

    const int size = (height+2)*(width+2); // grid size incl. boundaries
    const float dt = 0.0001; // incremental time step length

    float * u; // fluid field variables
    float * v;
    float * u_prev;
    float * v_prev;
    float * dens;
    float * dens_prev;
    bool * occupiedGrid; // define flow obstacles

    float visc = 0.001; // viscosity
    float diff = 0.01; // diffusion rate

    void initializeGrid(); // initialize grid variables
    void initializeFluid(); // initialize fluid velocities and density distribution at t=0

    void get_from_UI(float *dens, float *u, float *v, float *dens_prev, float *u_prev, float *v_prev); // static density input
    void vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt); // determine velocity vectors in next time step
    void dens_step(float *dens, float *dens_prev, float *u, float *v, float diff, float dt); // determine fluid field in next time step

    void project(float *u, float *v, float *u0, float *v0 );
    void add_source(float *x, float *x0, float dt );
    void set_bnd(int b, float *x);
    void diffuse(int,float *x,float *x0, float diff, float dt);
    void advect(int b, float * d, float * d0, float * u, float * v, float dt );

    // functions
    bool on_timeout(); //return true to keep the timeout and false to end it
    void update_view(float * dens);
    bool on_eventbox_button_press(GdkEventButton*);

    // printing
    void print_helper();

    Simulation();
    virtual ~Simulation();
};



#endif //PROJ_SIMULATION_H
