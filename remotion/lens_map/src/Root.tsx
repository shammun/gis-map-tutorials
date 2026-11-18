import React from "react";
import { Composition } from "remotion";
import { FPS, FRAMES, LensMapClip } from "./LensMapClip";

export const Root: React.FC = () => (
  <Composition id="LensMapClip" component={LensMapClip} durationInFrames={FRAMES} fps={FPS} width={1920} height={1080} />
);
