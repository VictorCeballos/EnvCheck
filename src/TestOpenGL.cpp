#include "inc/TestOpenGL.h"

GLdouble width = 600;
GLdouble height = 600;

std::string makeMeString(GLint versionRaw)
{
    std::stringstream ss;
    std::string str = "\0";
    ss << versionRaw;
    str = ss.str();
    return str;
}

void formatMe(std::string *text)
{
    std::string dot = ".";
    text->insert(1, dot);
    text->insert(4, dot);
}

void consoleMessage()
{
    char* versionGL = nullptr;
    GLint versionFreeGlutInt = 0;

    versionGL = (char *)(glGetString(GL_VERSION));
    versionFreeGlutInt = (glutGet(GLUT_VERSION));

    std::string versionFreeGlutString = makeMeString(versionFreeGlutInt);
    formatMe(&versionFreeGlutString);

    std::cout << std::endl;
    std::cout << "OpenGL version: " << versionGL << std::endl << std::endl;
    std::cout << "FreeGLUT version: " << versionFreeGlutString << std::endl << std::endl;

    std::cout << "GLEW version: " <<
                 GLEW_VERSION << "." << GLEW_VERSION_MAJOR << "." <<
                 GLEW_VERSION_MINOR << "." << GLEW_VERSION_MICRO << std::endl;
}

void managerError()
{
    if (glewInit())
    {
        std::cerr << "Unable to initialize GLEW." << std::endl;
        while (1);
        exit(EXIT_FAILURE);
    }
}

void managerDisplay(void)
{
    int ends[2][2];
    ends[0][0] = (int)(0.25*width);
    ends[0][1] = (int)(0.75*height);
    ends[1][0] = (int)(0.75*width);
    ends[1][1] = (int)(0.25*height);

    glClearColor(1.0, 1.0, 1.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0.0, 0.0, 0.0);
    glLineWidth(3.0);

    glBegin(GL_LINES);
    for (int i = 0; i < 2; ++i)
    {
        glVertex2iv((GLint *) ends[i]);
    }
    glEnd();

    glFlush();
    glutSwapBuffers();
}

void managerReshape(int w, int h)
{
    width = (GLdouble) w;
    height = (GLdouble) h;

    glViewport(0, 0, (GLsizei) w, (GLsizei) h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, w, 0.0, h);
}

void managerKeyboard(unsigned char key, int x, int y)
{
    if (key == 27) // 27 = ESC key
    {
        exit(0);
    }
}

void TestOpenGL::execute(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayString("");
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
    glutInitWindowSize((int) width, (int) height);
    //glutInitContextVersion(3, 3);
    //glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("Experiment with line drawing");

    managerError();
    consoleMessage();

    // Register the display callback function
    glutReshapeFunc(managerReshape);
    glutDisplayFunc(managerDisplay);
    glutKeyboardFunc(managerKeyboard);

    // Main loop
    glutMainLoop();
}

