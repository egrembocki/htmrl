import math
import random

from psu_capstone.agent_layer.our_htm.cell import Cell
from psu_capstone.agent_layer.our_htm.column import Column
from psu_capstone.agent_layer.our_htm.synapse import Synapse


class SpatialPooler:
    """
    Spatial Pooling algorithm steps
        1. Start with an input consisting of a fixed number of bits. These bits might represent sensory data or they might come
        from another region elsewhere in the HTM system.
        2. Initialize the HTM region by assigning a fixed number of columns to the region receiving this input. Each column has
        an associated dendritic segment, serving as the connection to the input space. Each dendrite segment has a set of
        potential synapses representing a (random) subset of the input bits. Each potential synapse has a permanence value.
        These values are randomly initialized around the permanence threshold. Based on their permanence values, some of the
        potential synapses will already be connected; the permanences are greater than than the threshold value.
        3. For any given input, determine how many connected synapses on each column are connected to active (ON) input bits.
        These are active synapses.
        4. The number of active synapses is multiplied by a "boosting" factor, which is dynamically determined by how often a
        column is active relative to its neighbors.
        5. A small percentage of columns within the inhibition radius with the highest activations (after boosting) become active,
        and disable the other columns within the radius. The inhibition radius is itself dynamically determined by the spread of
        input bits. There is now a sparse set of active columns.
        6. The region now follows the Spatial Pooling (Hebbian-style) learning rule: For each of the active columns, we adjust the
        permanence values of all the potential synapses. The permanence values of synapses aligned with active input bits are
        increased. The permanence values of synapses aligned with inactive input bits are decreased. The changes made to
        permanence values may change some synapses from being connected to unconnected, and vice-versa.
        7. For subsequent inputs, we repeat from step 3.

        Initialize the Spatial Pooler and run Phase 1 of the algorithm.

        Parameters follow the names and recommended defaults from Tables 1 and 2
        of the paper.

        Args:
            input_size: Number of bits in the input vector. The paper refers to
                this as the input space that columns can potentially connect to.
            column_count: Total number of columns in the HTM region. The paper
                recommends "a minimum value of 2048."
            potential_radius: This value will determine the spread of a column’s
                influence across the HTM layer. A small potential radius will keep a column’s receptive field local, while a very large potential
                radius will give the column global coverage over the input space.
            potential_pct: The percent of the inputs, within a column's potential radius, that are initialized to be in
                this column’s potential synapses. This should be set so that on average, at least 15-20
                input bits are connected when the Spatial Pooling algorithm is initialized. For example,
                suppose the input to a column typically contains 40 ON bits and that permanences are
                initialized such that 50% of the synapses are initially connected. In this case you will want
                potentialPct to be at least 0.75 since 40*0.5*0.75 = 15.
            global_inhibition: With global inhibition (globalInhibition=True), the most active columns are selected from the entire layer. Otherwise the winning
                columns are selected with respect to the columns’ local neighborhoods. The former offers a significant performance boost, and is
                often what we use in practice. With global inhibition turned off, the columnar inhibition takes effect in local neighborhoods.
            num_active_columns_per_inh_area: A parameter controlling the number of columns that will be winners after the inhibition
                step. We usually set this to be 2% of the expected inhibition radius. For 2048 columns and
                global inhibition, this is set to 40. We recommend a minimum value of 25.
            stimulus_threshold: A minimum number of inputs that must be active for a column to be considered during the
                inhibition step. This is roughly the background noise level expected out of the encoder and
                is often set to a very low value (0 to 5). The system is not very sensitive to this parameter;
                set to 0 if unsure.
            syn_perm_active_inc: Amount permanence values of active synapses are incremented during learning. This
                parameter is somewhat data dependent. (The amount of noise in the data will determine
                the optimal ratio between synPermActiveInc and synPermInactiveDec.) Usually set to a
                small value, such as 0.03
            syn_perm_inactive_dec: Amount permanence values of inactive synapses are decremented during learning. Usually
                set to a value smaller than the increment, such as 0.015.
            connected_perm:  the minimum permanence value at which a synapse is
                considered "connected".
            min_pct_overlap_duty_cycle: Before inhibition, if a
                column’s overlap duty cycle is below its minimum acceptable value (calculated dynamically as a function of
                minPctOverlapDutyCycle and the overlap duty cycle of neighboring columns), then all its permanence values are boosted by the
                increment amount. A subpar duty cycle implies either a column's previously learned inputs are no longer ever active, or the vast
                majority of them have been "hijacked" by other columns. By raising all synapse permanences in response to a subpar duty cycle
                before inhibition, we enable a column to search for new inputs.
            duty_cycle_period: Sliding window length (in iterations) over which
                the active and overlap duty cycles are averaged. The paper
                suggests "e.g. over the last 1000 iterations."
            boost_strength: Strength of the exponential boost function. A value
                of 0.0 disables boosting entirely.
            seed: RNG seed for reproducible potential-synapse initialization.
    """

    def __init__(
        self,
        input_size,
        column_count=2048,
        potential_radius=None,
        potential_pct=0.75,
        global_inhibition=True,
        num_active_columns_per_inh_area=40,
        stimulus_threshold=0,
        syn_perm_active_inc=0.03,
        syn_perm_inactive_dec=0.015,
        connected_perm=0.2,
        min_pct_overlap_duty_cycle=0.001,
        duty_cycle_period=1000,
        boost_strength=2.0,
        seed=42,
    ):

        self.input_size = input_size
        self.column_count = column_count

        self.potential_radius = potential_radius if potential_radius is not None else input_size
        self.potential_pct = potential_pct
        self.global_inhibition = global_inhibition
        self.num_active_columns_per_inh_area = num_active_columns_per_inh_area
        self.stimulus_threshold = stimulus_threshold
        self.syn_perm_active_inc = syn_perm_active_inc
        self.syn_perm_inactive_dec = syn_perm_inactive_dec
        self.connected_perm = connected_perm
        self.min_pct_overlap_duty_cycle = min_pct_overlap_duty_cycle
        self.duty_cycle_period = duty_cycle_period
        self.boost_strength = boost_strength

        self._rng = random.Random(seed)

        self.inhibition_radius = input_size

        self.iteration = 0

        self.columns: list[Column] = []
        for i in range(column_count):
            self.columns.append(Column(index=i))
        self._initialize_potential_synapses()

    def _initialize_potential_synapses(self):
        """
        Phase 1 of the algorithm: seed each column's potential synapses.

        From the paper:
            "Prior to receiving any inputs, the Spatial Pooling algorithm is
            initialized by computing a list of initial potential synapses for
            each column. This consists of a random set of inputs selected from
            the input space (within a column's inhibition radius). Each input
            is represented by a synapse and assigned a random permanence value.
            The random permanence values are chosen with two criteria. First,
            the values are chosen to be in a small range around connectedPerm,
            the minimum permanence value at which a synapse is considered
            'connected'. This enables potential synapses to become connected
            (or disconnected) after a small number of training iterations.
            Second, each column has a natural center over the input region,
            and the permanence values have a bias towards this center, so that
            they have higher values near the center."
        """
        for col in self.columns:
            center = int(col.index * (self.input_size / self.column_count))
            low = max(0, center - self.potential_radius)
            high = min(self.input_size, center + self.potential_radius + 1)

            window = []
            for i in range(low, high):
                window.append(i)

            num_potential = max(1, int(len(window) * self.potential_pct))
            chosen = self._rng.sample(window, num_potential)

            for src in chosen:
                perm = self._rng.uniform(
                    max(0.0, self.connected_perm - 0.1), min(1.0, self.connected_perm + 0.1)
                )
                syn = Synapse(src, perm)
                col.add_potential_synapse(syn, self.connected_perm)

    def compute(self, input_vector, learn=True):
        """
        Run one iteration of the Spatial Pooler on an input vector.

        Executes phases 2, 3, and optionally 4 of the algorithm in sequence,
        as described in the paper:
            "After initialization (phase 1), every iteration of the Spatial
            Pooling algorithm's compute routine goes through three distinct
            phases (phase 2 through phase 4) that occur in sequence."

        After inhibition has selected the winning columns, cell activations
        on each column are updated so that downstream consumers (such as a
        Temporal Memory layer built on top of the SP) can read activity at
        either the column or cell level.

        Args:
            input_vector: A sequence of 0/1 values of length `input_size`.
            learn: If True, Phase 4 runs and synapse permanences, boost
                values, duty cycles, and the inhibition radius are updated.
                If False, the SP operates in inference mode and is
                deterministic for a given input.

        Returns:
            List of column indices that became active after inhibition.

        Raises:
            ValueError: If the input vector length does not match `input_size`.
        """
        if len(input_vector) != self.input_size:
            raise ValueError(
                f"Input vector length {len(input_vector)} != expected {self.input_size}"
            )
        self.iteration += 1

        self._phase2_overlap(input_vector)
        active_columns = self._phase3_inhibition()

        if learn:
            self._phase4_learn(input_vector, active_columns)

        for col in self.columns:
            col.deactivate_cells()
        for c_idx in active_columns:
            self.columns[c_idx].activate_cells()

        return active_columns

    def _phase2_overlap(self, input_vector):
        """
        Phase 2: compute the overlap of each column with the current input.

        From the paper:
            "Given an input vector, this phase calculates the overlap of each
            column with that vector. The overlap for each column is simply
            the number of connected synapses with active inputs, multiplied
            by the column's boost factor."

        Pseudocode:
            for c in columns:
                overlap(c) = 0
                for s in connectedSynapses(c):
                    overlap(c) += input(t, s.sourceInput)
                overlap(c) *= boost(c)

        Each Column maintains its own `connected_synapses` list incrementally,
        so the inner loop reads from that list directly rather than filtering
        potential synapses by permanence on every call.
        Args:
            input_vector: A sequence of 0/1 values of length `input_size`.
        """
        for col in self.columns:
            col.compute_overlap(input_vector)

    def _phase3_inhibition(self):
        """
        Phase 3: select the winning columns after inhibition.

        From the paper:
            "The third phase calculates which columns remain as winners after
            the inhibition step. localAreaDensity is a parameter that controls
            the desired density of active columns within a local inhibition
            area. Alternatively, the density can be controlled by parameter
            numActiveColumnsPerInhArea. [...] The inhibition logic will ensure
            that at most numActiveColumnsPerInhArea columns become active in
            each local inhibition area. For example, if
            numActiveColumnsPerInhArea is 10, a column will be a winner if it
            has a non-zero overlap and its overlap score ranks 10th or higher
            among the columns within its inhibition radius."

        Pseudocode:
            for c in columns:
                minLocalActivity = kthScore(neighbors(c),
                                            numActiveColumnsPerInhArea)
                if overlap(c) > stimulusThreshold and
                   overlap(c) >= minLocalActivity then
                    activeColumns(t).append(c)

        Two paths are implemented: global inhibition, where winners are
        selected from the entire region at once, and local inhibition, where
        each column competes only against columns within its inhibition
        radius. The paper recommends global inhibition in practice for
        performance reasons.

        Returns:
            List of winning column indices, in ascending order.
        """
        active = []
        k = self.num_active_columns_per_inh_area

        if self.global_inhibition:
            sorted_cols = sorted(self.columns, key=lambda c: c.overlap, reverse=True)
            if len(sorted_cols) < k:
                threshold = 0
            else:
                threshold = sorted_cols[k - 1].overlap
            for col in self.columns:
                if (
                    col.overlap >= self.stimulus_threshold
                    and col.overlap >= threshold
                    and col.overlap > 0
                ):
                    active.append(col.index)
                    if len(active) >= k:
                        break
        else:
            for col in self.columns:
                neighbors = self._neighbors(col.index)

                # collect overlap values from neighbors, then sort descending.
                neighbor_overlaps = []
                for n in neighbors:
                    neighbor_overlaps.append(self.columns[n].overlap)
                neighbor_overlaps.sort(reverse=True)

                if len(neighbor_overlaps) < k:
                    min_local_activity = 0
                else:
                    min_local_activity = neighbor_overlaps[k - 1]
                if (
                    col.overlap > self.stimulus_threshold
                    and col.overlap >= min_local_activity
                    and col.overlap > 0
                ):
                    active.append(col.index)

        return active

    def _neighbors(self, column_index):
        """
        Return the indices of all columns within inhibition_radiu of the
        given column.

        From the paper's Table 1:
            "neighbors(c): A list of all the columns that are within
            inhibitionRadius of column c."

        The inhibition radius itself is recomputed at the end of each Phase 4
        as the average connected receptive field size across all columns, so
        the size of a column's neighborhood adapts as learning proceeds.
        Args:
            column_index: the index of the column to find neighbors for.
        """
        low = max(0, column_index - self.inhibition_radius)
        high = min(self.column_count, column_index + self.inhibition_radius + 1)

        neighbors = []
        for i in range(low, high):
            neighbors.append(i)
        return neighbors

    def _phase4_learn(self, input_vector, active_columns):
        """
        Phase 4: update synapse permanences, duty cycles, boost values, and
        the inhibition radius.

        From the paper:
            "This final phase performs learning, updating the permanence
            values of all synapses as necessary, as well as the boost values
            and inhibition radii. The main learning rule is implemented in
            lines 14-20. For winning columns, if a synapse is active, its
            permanence value is incremented, otherwise it is decremented;
            permanence values are constrained to be between 0 and 1. Notice
            that permanence values on synapses of non-winning columns are
            not modified."

        Boosting has two independent mechanisms, both described in the paper:
            "If a column does not win often enough (as measured by
            activeDutyCycle) compared to its neighbors, its overall boost
            value is set to be greater than 1. If a column is active more
            frequently than its neighbors, its overall boost value is set to
            be less than one."
            "If a column's connected synapses do not overlap well with any
            inputs often enough (as measured by overlapDutyCycle), its
            permanence values are boosted."
        Args:
            input_vector: A sequence of 0/1 values of length `input_size`.
            active_columns: The columns that are active.

        """
        cp = self.connected_perm
        inc = self.syn_perm_active_inc
        dec = self.syn_perm_inactive_dec

        for c_idx in active_columns:
            col = self.columns[c_idx]
            for syn in col.potential_synapses:
                if input_vector[syn.source_input_index] == 1:
                    crossing = syn.increment_permanence(inc, cp)
                else:
                    crossing = syn.decrement_permanence(dec, cp)
                if crossing:
                    col.on_crossing(syn, crossing)

        active_set = set(active_columns)
        for col in self.columns:
            self._update_active_duty_cycle(col, col.index in active_set)
            self._update_overlap_duty_cycle(col)

        if self.global_inhibition:
            total_adc = 0.0
            max_odc = 0.0
            for col in self.columns:
                total_adc += col.active_duty_cycle
                if col.overlap_duty_cycle > max_odc:
                    max_odc = col.overlap_duty_cycle
            mean_neighbor_active = total_adc / self.column_count
            min_duty_cycle = self.min_pct_overlap_duty_cycle * max_odc

            for col in self.columns:
                col.boost = self._boost_function(col.active_duty_cycle, mean_neighbor_active)
                if col.overlap_duty_cycle < min_duty_cycle:
                    self._increase_permanences(col, 0.1 * self.connected_perm)
        else:
            for col in self.columns:
                neighbors = self._neighbors(col.index)

                # sum active duty cycles across neighbors to get the mean.
                total_neighbor_adc = 0.0
                for n in neighbors:
                    total_neighbor_adc += self.columns[n].active_duty_cycle
                mean_neighbor_active = total_neighbor_adc / max(1, len(neighbors))

                col.boost = self._boost_function(col.active_duty_cycle, mean_neighbor_active)

                # find the max overlap duty cycle across neighbors.
                max_neighbor_overlap_dc = 0.0
                for n in neighbors:
                    neighbor_odc = self.columns[n].overlap_duty_cycle
                    if neighbor_odc > max_neighbor_overlap_dc:
                        max_neighbor_overlap_dc = neighbor_odc

                min_duty_cycle = self.min_pct_overlap_duty_cycle * max_neighbor_overlap_dc
                if col.overlap_duty_cycle < min_duty_cycle:
                    self._increase_permanences(col, 0.1 * self.connected_perm)

        self.inhibition_radius = self._average_receptive_field_size()

    def _update_active_duty_cycle(self, col, c_active):
        """
        Update a column's active duty cycle with the current iteration's
        activation.

        From the paper's Table 1:
            "activeDutyCycle(c): A sliding average representing how often
            column c has been active after inhibition (e.g. over the last
            1000 iterations)."

        Implemented as an exponential moving average with a warm-up period
        equal to `duty_cycle_period`: during the first `duty_cycle_period`
        iterations the effective window grows with the iteration count, so
        early measurements are not dominated by the zero initial value.
        Args:
            col: The column to update duty cycle.
            c_active: if the column is active or not.
        """
        period = min(self.iteration, self.duty_cycle_period)
        col.active_duty_cycle = (
            col.active_duty_cycle * (period - 1) + (1.0 if c_active else 0.0)
        ) / period

    def _update_overlap_duty_cycle(self, col):
        """
        Update a column's overlap duty cycle.

        From the paper's Table 1:
            "overlapDutyCycle(c): A sliding average representing how often
            column c has had significant overlap (i.e. greater than
            stimulusThreshold) with its inputs (e.g. over the last 1000
            iterations)."

        A "significant overlap" here means the column's boosted overlap is
        both strictly greater than zero and at or above the stimulus
        threshold for this iteration.
        Args:
            col: the column to update overlap duty cycle.
        """
        period = min(self.iteration, self.duty_cycle_period)
        had_overlap = 1.0 if col.overlap >= self.stimulus_threshold and col.overlap > 0 else 0.0
        col.overlap_duty_cycle = (col.overlap_duty_cycle * (period - 1) + had_overlap) / period

    def _boost_function(self, active_duty_cycle, neighbor_mean_duty_cycle):
        """
        Compute the boost factor for a column.

        From the paper:
            "The boostFunction is an exponential function that depends on
            the difference between the active duty cycle of a column and
            the average active duty cycles of its neighbors."

        Form used:
            boost = exp(-boost_strength * (active_duty_cycle - mean_neighbors))

        When the column's duty cycle is below the mean, the exponent is
        positive and boost > 1 (the column is encouraged to win). When it
        is above the mean, boost < 1 (the column is discouraged). When
        `boost_strength` is 0, boost is always 1 and boosting is disabled.
        Args:
            active_duty_cycle: the columns active duty cycle.
            neighbor_mean_duty_cycle: the columns surrounding the column average duty cycles.
        """
        return math.exp(-self.boost_strength * (active_duty_cycle - neighbor_mean_duty_cycle))

    def _increase_permanences(self, col, scale):
        """
        Raise all permanences in a column by a scalar amount.

        Used by Phase 4's overlap-duty-cycle boosting path. From the paper:
            "If a column's overlap duty cycle is below its minimum
            acceptable value [...] then all its permanence values are
            boosted by the increment amount. A subpar duty cycle implies
            either a column's previously learned inputs are no longer ever
            active, or the vast majority of them have been 'hijacked' by
            other columns. By raising all synapse permanences in response
            to a subpar duty cycle before inhibition, we enable a column to
            search for new inputs."

        Because this can push permanences across the connected threshold,
        crossings are reported to the column so its `connected_synapses`
        list stays up to date.
        Args:
            col: the column to increase permanence for.
            scale: adjust permanence changes
        """
        cp = self.connected_perm
        for syn in col.potential_synapses:
            crossing = syn.increment_permanence(scale, cp)
            if crossing:
                col.on_crossing(syn, crossing)

    def _average_receptive_field_size(self):
        """
        Compute the average connected receptive field radius across all
        columns.

        From the paper's Table 2:
            "averageReceptiveFieldSize(): The radius of the average
            connected receptive field size of all the columns. The
            connected receptive field size of a column includes only the
            connected synapses (those with permanence values >=
            connectedPerm). This is used to determine the extent of lateral
            inhibition between columns."

        For each column with at least one connected synapse, the receptive
        field radius is taken to be half the span of its connected source
        indices (max - min divided by two). Columns with no connected
        synapses are skipped. If no column has any connected synapses, the
        full input size is returned as a safe fallback.
        """
        total = 0
        n = 0
        for col in self.columns:
            if not col.connected_synapses:
                continue

            # collect source indices of all connected synapses
            indices = []
            for s in col.connected_synapses:
                indices.append(s.source_input_index)

            radius = (max(indices) - min(indices)) / 2.0
            total += radius
            n += 1
        if n == 0:
            return self.input_size
        return max(1, int(total / n))
