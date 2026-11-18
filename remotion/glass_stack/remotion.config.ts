// Render settings of the glass-pane stack film. Flags on the command line override them.
import { Config } from "@remotion/cli/config";

Config.setChromiumOpenGlRenderer("angle"); // WebGL for three.js (on Linux without a GPU pass --gl=swangle instead)
Config.setVideoImageFormat("jpeg");
Config.setJpegQuality(90);
Config.setCodec("h264");
Config.setCrf(23);
Config.setPixelFormat("yuv420p");
Config.setColorSpace("bt709"); // tv-range yuv420p that every browser plays with the right colours
Config.setConcurrency("50%");
Config.setOverwriteOutput(true);
