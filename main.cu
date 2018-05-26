#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <gtkmm.h>

#include "simulation.h"
#include "numerical_kernels.h"


using namespace std;


int main(int argc, char* argv[])
{



    // open GUI window
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "continuum");
    
    // initiate simulation instance
    Simulation sim;
    
    
    //Run the app with a simulation
    return app->run(sim);
}


