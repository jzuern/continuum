#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <gtkmm.h>

#include "simulation.h"
#include "numerical_kernels.h"
#include "numerical_kernels.cu"


using namespace std;



int main(int argc, char* argv[])
{

    // open GUI window
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "continuum.de");
    
    // initiate simulation instance
    Simulation sim;

    sim.init();
    
    
    //Run the app with a simulation
    return app->run(sim);
}


