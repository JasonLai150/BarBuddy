export type LiftType = 'Bench Press' | 'Squat' | 'Deadlift' | 'Overhead Press' | 'Barbell Row' | 'Dumbbell Curl' | 'Pull-ups';

export interface Lift {
  id: string;
  type: LiftType;
  videoUri?: string;
  videoFilename?: string;
  date: Date;
}

export const LIFT_TYPES: LiftType[] = [
  'Bench Press',
  'Squat',
  'Deadlift',
  'Overhead Press',
  'Barbell Row',
  'Dumbbell Curl',
  'Pull-ups',
];
