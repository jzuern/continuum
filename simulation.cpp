//
// Created by jannik on 5/17/18.
//
#include <gtkmm.h>
#include <iostream>
#include "simulation.h"
#include <cmath>        // std::abs


using namespace std;


Simulation::Simulation(){

    width = 500;
    height = 500;

    set_default_size(width, height);
    set_title("Gtkmm Programming - C++");
    set_position(Gtk::WIN_POS_CENTER);

    add(img);

    time_step_counter = 0;


    int rowstride = width*3;
    bool has_alpha = false;
    int bits_per_sample = 8;

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
            if (i > middle_x and j > middle_y)
            {
                data_array_new[i][j] = 1.0f;
            }
        }
    }

    update_view(data_array_new);


//    Glib::RefPtr<Gdk::Pixbuf> ref_orig = Gdk::Pixbuf::create_from_file ("/home/jannik/Desktop/green.jpg");
//
//    guchar * data = ref_orig->get_pixels();

    //create slot for timeout signal
    int timeout_value = 100; //in ms
    sigc::slot<bool>my_slot = sigc::mem_fun(*this, &Simulation::on_timeout);

    //connect slot to signal
    Glib::signal_timeout().connect(my_slot, timeout_value);

    show_all_children();

}

Simulation::~Simulation(){


}




bool Simulation::on_timeout(){

    time_step_counter += 1;

    // update old_data
    data_array_old = data_array_new;


    int middle_x = width / 2;
    int middle_y = height / 2;


    // populate data array
    for (int i = 1; i < width-1; i++)
    {
        for (int j = 1; j < height-1; j++)
        {

            float lower = data_array_new[i][j-1];
            float upper = data_array_new[i][j+1];
            float left = data_array_new[i-1][j];
            float right = data_array_new[i+1][j];

            float clf = 0.42; // must be smaller than 0.5, otherwise instable!

            data_array_new[i][j] = data_array_old[i][j] + clf*(lower + upper + left + right - 4*data_array_old[i][j]);
        }
    }

    // update view of data array
    update_view(data_array_new);




    return true;
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

