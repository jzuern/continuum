#include <iostream>

#include <jpeglib.h>
#include <gtkmm.h>
#include "simulation.h"


unsigned char * read_image(const char* Name){


    unsigned char a, r, g, b;
    int width, height;
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    FILE * infile;        /* source file */
    JSAMPARRAY pJpegBuffer;       /* Output row buffer */
    int row_stride;       /* physical row width in output buffer */

    if ((infile = fopen(Name, "rb")) == NULL) {
        fprintf(stderr, "can't open %s\n", Name);
    }
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    (void) jpeg_read_header(&cinfo, TRUE);
    (void) jpeg_start_decompress(&cinfo);
    width = cinfo.output_width;
    height = cinfo.output_height;

    unsigned char * pDummy = new unsigned char [width*height*4];

    row_stride = width * cinfo.output_components;
    pJpegBuffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
        for (int x = 0; x < width; x++) {
            a = 0; // alpha value is not supported on jpg
            r = pJpegBuffer[0][cinfo.output_components * x];
            if (cinfo.output_components > 2) {
                g = pJpegBuffer[0][cinfo.output_components * x + 1];
                b = pJpegBuffer[0][cinfo.output_components * x + 2];
            } else {
                g = r;
                b = r;
            }
            *(pDummy++) = b;
            *(pDummy++) = g;
            *(pDummy++) = r;
            *(pDummy++) = a;
        }
    }
    fclose(infile);
    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return pDummy;

}



//int main(int argc, char * argv[]) {
//
//    std::cout << "Hello, World!" << std::endl;
//
//
////    printf("reading jpg...\n");
////    char fname[] = "/home/jannik/Desktop/1.jpeg";
////    unsigned char * image_array;
////    image_array = read_image(fname);
////    printf("...done!\n");
////
////
////
////    auto app = Gtk::Application::create(argc, argv,"org.gtkmm.examples.base");
////
////    Gtk::Window window;
////    window.set_default_size(500, 500);
////
////    return app->run(window);
//
//
//    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create();
//
//    Gtk::Window main_window;
//
//    main_window.set_title ("Show Image");
//    main_window.set_position (Gtk::WIN_POS_CENTER);
//
//    Glib::RefPtr<Gdk::Pixbuf> ref_orig = Gdk::Pixbuf::create_from_file ("/home/jannik/Desktop/1.jpeg");
//
//    const guint8* raw = ref_orig->get_pixels();
//    const std::vector<guint8> image_pixels (raw, raw + ref_orig->get_byte_length());
//
//    Glib::RefPtr<Gdk::Pixbuf> ref_dest =
//            Gdk::Pixbuf::create_from_data (
//                    image_pixels.data(), Gdk::COLORSPACE_RGB,
//                    ref_orig->get_has_alpha(), ref_orig->get_bits_per_sample(),
//                    ref_orig->get_width(), ref_orig->get_height(), ref_orig->get_rowstride());
//
//    Gtk::Image image;
//    image.set (ref_dest);
//
//    main_window.add(image);
//
//    main_window.show_all_children();
//
//    return app->run (main_window);
//
//}



using namespace std;


int main(int argc, char* argv[])
{
    Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "com.kaze.test");

    Simulation sim;

    // The Gui Window is displayed
    return app->run(sim);
}

