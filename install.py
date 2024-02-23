import launch

if not launch.is_installed("pillow"):
    launch.run_pip("install pillow",
                   "requirements for RealImageArtifacts")
if not launch.is_installed("numpy"):
    launch.run_pip("install numpy",
                   "requirements for RealImageArtifacts")
if not launch.is_installed("piexif"):
    launch.run_pip("install piexif",
                   "requirements for RealImageArtifacts")
