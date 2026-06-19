"use client";

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";
import { type PredictV2Response, type PredictRequest, type DrugHint } from "@/lib/api";

interface PredictionState {
  request: PredictRequest | null;
  response: PredictV2Response | null;
  drug: DrugHint | null;
  loading: boolean;
  error: string | null;
  continualLearning: boolean;
  modelUpdated: boolean;
  sessionLabel: string | null;
  predictionGeneratedAt: string | null;
}

interface PredictionContextValue extends PredictionState {
  setRequest: (req: PredictRequest) => void;
  setResponse: (res: PredictV2Response) => void;
  setDrug: (d: DrugHint | null) => void;
  setLoading: (v: boolean) => void;
  setError: (msg: string | null) => void;
  setSessionMeta: (sessionLabel: string, predictionGeneratedAt: string) => void;
  toggleContinualLearning: () => void;
  markModelUpdated: () => void;
  isSafe: boolean;
}

const PredictionContext = createContext<PredictionContextValue | null>(null);

export function PredictionProvider({ children }: { children: ReactNode }) {
  const [request, setRequest] = useState<PredictRequest | null>(null);
  const [response, setResponse] = useState<PredictV2Response | null>(null);
  const [drug, setDrug] = useState<DrugHint | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [continualLearning, setContinualLearning] = useState(false);
  const [modelUpdated, setModelUpdated] = useState(false);
  const [sessionLabel, setSessionLabel] = useState<string | null>(null);
  const [predictionGeneratedAt, setPredictionGeneratedAt] = useState<string | null>(null);

  const toggleContinualLearning = useCallback(() => setContinualLearning((v) => !v), []);
  const markModelUpdated = useCallback(() => setModelUpdated(true), []);
  const setSessionMeta = useCallback((label: string, at: string) => {
    setSessionLabel(label);
    setPredictionGeneratedAt(at);
  }, []);

  const isSafe = response ? response.safety.is_safe : true;

  return (
    <PredictionContext.Provider
      value={{
        request,
        response,
        drug,
        loading,
        error,
        continualLearning,
        modelUpdated,
        sessionLabel,
        predictionGeneratedAt,
        setRequest,
        setResponse,
        setDrug,
        setLoading,
        setError,
        setSessionMeta,
        toggleContinualLearning,
        markModelUpdated,
        isSafe,
      }}
    >
      {children}
    </PredictionContext.Provider>
  );
}

export function usePrediction() {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error("usePrediction must be used within PredictionProvider");
  return ctx;
}
