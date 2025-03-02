#include "Image_Processing.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Image_Processing w;
    w.show();
    return a.exec();
}
