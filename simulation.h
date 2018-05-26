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


    // i -> Index right/left
    // j -> Index up/down
    // u -> velocity right/left
    // v -> velocity up/down

    // GUI members
    Gtk::Image img;
    Gtk::EventBox box;
    guint8 * img_data;

    // simulation stuff
    int time_step_counter = 0;
    const float dt = 0.0001; // incremental time step length
    const int height = 200;
    const int width = 200;
    const int size = (height+2)*(width+2); // grid size incl. boundaries
    const int N = height;

    // numerical parameters
    int gauss_seidel_iterations = 10;




    void initializeGrid(); // initialize grid variables
    void initializeFluid(); // initialize fluid velocities and density distribution at t=0

    // fluid dynamics routines

    void vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt); // determine velocity vectors in next time step
    void dens_step(float *dens, float *dens_prev, float *u, float *v, float diff, float dt); // determine fluid field in next time step

    void project(float *u, float *v, float *u0, float *v0 );
    void add_source(float *x, float *x0, float dt );
    void set_bnd(int b, float *x);
    void diffuse(int,float *x,float *x0, float diff, float dt);
    void advect(int b, float * d, float * d0, float * u, float * v, float dt );

    // GUI handling functions
    bool on_timeout(); //return true to keep the timeout and false to end it
    void update_view(float * dens);
    bool on_eventbox_button_press(GdkEventButton*);

    // printing
    void print_helper();

    // constructor
    Simulation();

    // destructor
    virtual ~Simulation();

private:



    float * u; // fluid field variables
    float * v;
    float * u_prev;
    float * v_prev;
    float * dens;
    float * dens_prev;
    bool * occupiedGrid; // define flow obstacles

    float visc = 0.001; // viscosity
    float diff = 0.01; // diffusion rate

};



#endif //PROJ_SIMULATION_H
