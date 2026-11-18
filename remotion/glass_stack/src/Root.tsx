import React from "react";
import { Composition } from "remotion";
import { FPS, FRAMES, GlassStack } from "./GlassStack";

export const Root: React.FC = () => (
  <Composition id="GlassStack" component={GlassStack} durationInFrames={FRAMES} fps={FPS} width={1920} height={1080} />
);
