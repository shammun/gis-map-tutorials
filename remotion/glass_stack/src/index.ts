// Entry point: npx remotion render src/index.ts GlassStack out/glass_stack_remotion.mp4 --public-dir=public
import { registerRoot } from "remotion";
import { Root } from "./Root";

registerRoot(Root);
