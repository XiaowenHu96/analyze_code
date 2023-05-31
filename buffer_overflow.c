#include <stdio.h>
#include <string.h>

#define BUFFER_SIZE 10

void copy_to_buffer(char *input) {
    char buffer[BUFFER_SIZE];
    strcpy(buffer, input);
    printf("Buffer contains: %s\n", buffer);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Please provide one argument!\n");
        return 1;
    }

    copy_to_buffer(argv[1]);
    return 0;
}
