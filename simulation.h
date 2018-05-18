//
// Created by jannik on 5/17/18.
//

#ifndef PROJ_SIMULATION_H
#define PROJ_SIMULATION_H


using namespace std;

class Simulation : public Gtk::Window{
public:
    Gtk::Image img;
    Gtk::EventBox box;

    int time_step_counter;

    int height;
    int width;

    guint8 * img_data;

    float ** data_array_old;
    float ** data_array_new;
    float ** velocity_array;

    bool on_timeout(); //return true to keep the timeout and false to end it
    void update_view(float ** data_array_new);
    bool on_eventbox_button_press(GdkEventButton*);

    Simulation();
    virtual ~Simulation();
};


#endif //PROJ_SIMULATION_H
