function time_feats = extract_time_hrv(rr_clean)
    % EXTRACT_TIME_HRV Computes 5 time-domain HRV features for a 1-minute segment of RR intervals.
    %
    % Features:
    %   1. mRR     - Mean NN interval (seconds)
    %   2. SDNN    - Standard deviation of NN intervals (seconds)
    %   3. RMSSD   - Root mean square of successive NN differences (seconds)
    %   4. pNN50   - Percentage of successive NN intervals differing by > 50 ms
    %   5. HR_mean - Mean heart rate (beats per minute, bpm)
    %
    % Input:
    %   rr_clean   - Vector of cleaned NN intervals in seconds for a 1-minute segment
    % Output:
    %   time_feats - 1x5 vector of extracted features [mRR, SDNN, RMSSD, pNN50, HR_mean]

    % Filter out NaNs (ectopic beats)
    rr = rr_clean(~isnan(rr_clean));

    if length(rr) < 3
        % Insufficient beats to compute reliable statistics
        time_feats = [NaN, NaN, NaN, NaN, NaN];
        return;
    end

    mRR = mean(rr);
    SDNN = std(rr);

    delta_rr = diff(rr);
    if isempty(delta_rr)
        RMSSD = NaN;
        pNN50 = NaN;
    else
        RMSSD = sqrt(mean(delta_rr .^ 2));
        pNN50 = sum(abs(delta_rr) > 0.050) / length(delta_rr);
    end

    HR_mean = 60 / mRR;

    time_feats = [mRR, SDNN, RMSSD, pNN50, HR_mean];
end
