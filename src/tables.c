#include "network.h"
#include "utils.h"
#include "parser.h"
/*
void write_outputs_to_file(float *outputVals, int outputSize, char *outputFileName){
    FILE *fout = fopen(outputFileName, "w");
    if(!fout) file_error(outputFileName);
    int predIdx = 0;
    for(predIdx=0;predIdx<outputSize;predIdx++){
        fwrite(outputVals[predIdx], sizeof(float), 1, fout);
    }
    fclose(fout);
}
*/

void train_tables(char *cfgfile, char *weightfile, char *dataFolder, char *labelsFolder)
{
    data_seed = time(0);
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 256;
    int i = net.seen/imgs;
    char trainListBuff[256];
    sprintf(trainListBuff, "%s/train.list", dataFolder);
    printf("Training list file: %s", trainListBuff);
    list *plist = get_paths(trainListBuff);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_tables(paths, imgs, plist->size, 256, 256, dataFolder, labelsFolder);
        printf("%d: Loaded partial training data = %d images\n", i, net.seen);

        float loss = train_network(net, train);
        printf("%d: Trained network\n");

        #ifdef GPU
        float *out = get_network_output_gpu(net);
        #else
        float *out = get_network_output(net);
        #endif
        // image pred = float_to_image(32, 32, 1, out);
        // print_image(pred);

        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if((i % 20000) == 0) net.learning_rate *= .1;
        //if(i%100 == 0 && net.learning_rate > .00001) net.learning_rate *= .97;
        if(i%250==0){
            char buff[256];
            sprintf(buff, "weights/%s_%d.weights", base, i);
            save_weights(net, buff);
        }
    }
}

void test_tables(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char filename[256];

    fgets(filename, 256, stdin);
    strtok(filename, "\n");
    image im = load_image_color(filename, 0, 0);
    //image im = load_image_color("/home/pjreddie/darknet/data/figs/C02-1001-Figure-1.png", 0, 0);
    image sized = resize_image(im, 256, 256);
    printf("%d %d %d\n", im.h, im.w, im.c);
    float *X = sized.data;
    time=clock();
    float *predictions = network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));

    // Write predicitons to output file
    int netOutputSize = get_network_output_size(net);
    char *outFileName = find_replace(filename, ".png", "-output.txt");
    printf("Writing predictions to the output file %s", outFileName);
    write_outputs_to_file(predictions, netOutputSize, outFileName);

    // Write last but 1 layer to output file
    float *lastButOneOutputs = get_network_output_lastButOne(net);
    int lboNetOutputSize = get_network_output_size_lastButOne(net);
    char *lboFileName = find_replace(filename, ".png", "-lastButOne.txt");
    printf("Writing the last but one layer outputs to the output file %s", lboFileName);
    write_outputs_to_file(lastButOneOutputs, lboNetOutputSize, lboFileName);

	free_image(im);
	free_image(sized);
}

void run_tables(int argc, char **argv)
{
    if(argc < 6){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [dataFolderName] [labelsFolderName] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *dataFolder = argv[4];
    char *labelsFolder = argv[5];
    char *weights = (argc > 6) ? argv[6] : 0;

    if(0==strcmp(argv[2], "train")) train_tables(cfg, weights, dataFolder, labelsFolder);
    if(0==strcmp(argv[2], "test")) test_tables(cfg, weights);
}
