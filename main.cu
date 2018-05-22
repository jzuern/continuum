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
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "continuum");
    Simulation sim;
    //The Gui Window is displayed
    return app->run(sim);
}


