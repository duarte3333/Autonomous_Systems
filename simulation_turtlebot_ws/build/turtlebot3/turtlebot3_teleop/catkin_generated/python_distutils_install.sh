#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/duarte/Documents/simulation_turtlebot/src/turtlebot3/turtlebot3_teleop"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/duarte/Documents/simulation_turtlebot/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/duarte/Documents/simulation_turtlebot/install/lib/python3/dist-packages:/home/duarte/Documents/simulation_turtlebot/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/duarte/Documents/simulation_turtlebot/build" \
    "/usr/bin/python3" \
    "/home/duarte/Documents/simulation_turtlebot/src/turtlebot3/turtlebot3_teleop/setup.py" \
     \
    build --build-base "/home/duarte/Documents/simulation_turtlebot/build/turtlebot3/turtlebot3_teleop" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/duarte/Documents/simulation_turtlebot/install" --install-scripts="/home/duarte/Documents/simulation_turtlebot/install/bin"
