"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

type ProgressState = {
  currentChapter: number;
  unlocked: number;
  completed: string[];
  reducedMotion: boolean;
  setCurrentChapter: (chapter: number) => void;
  completeChapter: (chapterId: string, chapterIndex: number) => void;
  setReducedMotion: (value: boolean) => void;
  resetProgress: () => void;
};

export const useProgressStore = create<ProgressState>()(
  persist(
    (set) => ({
      currentChapter: 0,
      unlocked: 0,
      completed: [],
      reducedMotion: false,
      setCurrentChapter: (currentChapter) => set({ currentChapter }),
      completeChapter: (chapterId, chapterIndex) =>
        set((state) => ({
          completed: state.completed.includes(chapterId)
            ? state.completed
            : [...state.completed, chapterId],
          unlocked: Math.max(state.unlocked, Math.min(7, chapterIndex + 1)),
        })),
      setReducedMotion: (reducedMotion) => set({ reducedMotion }),
      resetProgress: () =>
        set({
          currentChapter: 0,
          unlocked: 0,
          completed: [],
        }),
    }),
    {
      name: "ml-atlas-progress-v1",
      partialize: (state) => ({
        currentChapter: state.currentChapter,
        unlocked: state.unlocked,
        completed: state.completed,
        reducedMotion: state.reducedMotion,
      }),
    },
  ),
);
