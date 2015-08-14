#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "parser.h"
#include "utils.h"
#include "cuda.h"

#define _GNU_SOURCE
#include <fenv.h>

extern void run_imagenet(int argc, char **argv);
extern void run_detection(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_tables(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);

void change_rate(char *filename, float scale, float add)
{
    // Ready for some weird shit??
    FILE *fp = fopen(filename, "r+b");
    if(!fp) file_error(filename);
    float rate = 0;
    fread(&rate, sizeof(float), 1, fp);
    printf("Scaling learning rate from %f to %f\n", rate, rate*scale+add);
    rate = rate*scale + add;
    fseek(fp, 0, SEEK_SET);
    fwrite(&rate, sizeof(float), 1, fp);
    fclose(fp);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(&net, weightfile, max);
    }
    net.seen = 0;
    save_weights_upto(net, outfile, max);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rescale_filters(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            rgbgr_filters(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    visualize_network(net);
    #ifdef OPENCV
    cvWaitKey(0);
    #endif
}

int main(int argc, char **argv)
{
    //test_resize("data/bad.jpg");
    //test_box();
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) gpu_index = -1;

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cudaSetDevice(gpu_index);
    }
#endif

    if(0==strcmp(argv[1], "imagenet")){
        run_imagenet(argc, argv);
    } else if (0 == strcmp(argv[1], "detection")){
        run_detection(argc, argv);
    } else if (0 == strcmp(argv[1], "coco")){
        run_coco(argc, argv);
    } else if (0 == strcmp(argv[1], "dice")){
        run_dice(argc, argv);
    } else if (0 == strcmp(argv[1], "writing")){
        run_writing(argc, argv);
    } else if (0 == strcmp(argv[1], "tables")){
        run_tables(argc, argv);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "captcha")){
        run_captcha(argc, argv);
    } else if (0 == strcmp(argv[1], "nightmare")){
        run_nightmare(argc, argv);
    } else if (0 == strcmp(argv[1], "change")){
        change_rate(argv[2], atof(argv[3]), (argc > 4) ? atof(argv[4]) : 0);
    } else if (0 == strcmp(argv[1], "rgbgr")){
        rgbgr_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "rescale")){
        rescale_net(argv[2], argv[3], argv[4]);
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else if (0 == strcmp(argv[1], "imtest")){
        test_resize(argv[2]);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}
