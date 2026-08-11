export type KernelViewMode = "heatmap" | "signed_logo" | "activation_logo";
export type ChannelOrder = "original" | "stem_occupancy" | "final_rms" | "profile_influence" | "count_influence";
export type KernelOrientation = "model" | "reverse_complement";

export type ChannelMetric = {
  id_zero_based: number;
  label: string;
  checkpoint_id: string;
  stem_occupancy: number;
  final_rms: number;
  profile_influence: number;
  count_influence: number;
  final_mean_activation: number;
  count_weight: number;
  count_contribution: number;
};

export type ChannelRegistry = {
  channel_registry: ChannelMetric[];
  channel_orders: Record<ChannelOrder, number[]>;
};

export type ActivationMotif = {
  filter_id_zero_based: number;
  pfm_positions_by_bases: number[][];
  background_frequencies: number[];
  information_content_bits: number[];
  site_count: number;
  corpus_id: string;
  activation_selection_rule: string;
  orientation: KernelOrientation;
};

export const CHANNEL_ORDER_LABELS: Record<ChannelOrder, string> = {
  original: "Original model order",
  stem_occupancy: "Stem activation occupancy",
  final_rms: "Final-layer RMS activation",
  profile_influence: "Empirical profile influence",
  count_influence: "Empirical count influence",
};

export function centeredStemWeights(weightsBaseByPosition: number[][], bias: number) {
  const positionMeans = weightsBaseByPosition[0].map((_, position) => (
    weightsBaseByPosition.reduce((sum, row) => sum + row[position], 0) / 4
  ));
  return {
    centered: weightsBaseByPosition.map(row => row.map((value, position) => value - positionMeans[position])),
    adjustedBias: bias + positionMeans.reduce((sum, value) => sum + value, 0),
    removedPositionMeans: positionMeans,
  };
}

export function reverseComplementBaseRows(matrix: number[][]) {
  const complementRows = [3, 2, 1, 0];
  return complementRows.map(sourceRow => [...matrix[sourceRow]].reverse());
}

export function reverseComplementSequence(sequence: string) {
  const complement: Record<string, string> = { A: "T", C: "G", G: "C", T: "A", N: "N" };
  return [...sequence].reverse().map(base => complement[base] ?? "N").join("");
}

export function informationContent(pfmPositionsByBases: number[][], background = [.25, .25, .25, .25]) {
  void background;
  return pfmPositionsByBases.map(column => 2 + column.reduce((entropyTerm, probability) => (
    probability > 0 ? entropyTerm + probability * Math.log2(probability) : entropyTerm
  ), 0));
}

export function displayRank(order: number[], channelId: number) {
  return order.indexOf(channelId) + 1;
}

export function reorderValues<T>(values: T[], order: number[]) {
  return order.map(channel => values[channel]);
}
