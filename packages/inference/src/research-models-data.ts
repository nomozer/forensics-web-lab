import bundledData from './research-models-bundle.json' with { type: 'json' };
import { FoldCandidateModel, FoldParams } from './research-scorer.js';

export const RESEARCH_CANDIDATE_MODELS: FoldCandidateModel[] = (
  bundledData.outer_folds as FoldParams[]
).map((p) => new FoldCandidateModel(p));
