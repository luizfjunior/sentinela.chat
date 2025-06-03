
import { X, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface AudioRecorderProps {
  recordingTime: number;
  onCancel: () => void;
  onSend: () => void;
}

const AudioRecorder = ({ recordingTime, onCancel, onSend }: AudioRecorderProps) => {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex items-center gap-3 px-4 py-3 bg-red-50 dark:bg-red-900/20 rounded-2xl border border-red-200 dark:border-red-800">
      {/* Recording indicator */}
      <div className="flex items-center gap-2">
        <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
        <span className="text-sm font-medium text-red-600 dark:text-red-400">
          {formatTime(recordingTime)}
        </span>
      </div>

      {/* Recording wave animation */}
      <div className="flex items-center gap-1 flex-1 justify-center">
        {[1, 2, 3, 4, 5].map((i) => (
          <div
            key={i}
            className="w-1 bg-red-400 rounded-full animate-pulse"
            style={{
              height: `${Math.random() * 20 + 10}px`,
              animationDelay: `${i * 0.1}s`,
              animationDuration: '0.8s'
            }}
          ></div>
        ))}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-gray-500 hover:text-red-600 hover:bg-red-100 dark:hover:bg-red-900/30"
          onClick={onCancel}
        >
          <X className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          size="icon"
          className="h-8 w-8 bg-red-500 hover:bg-red-600 text-white"
          onClick={onSend}
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};

export default AudioRecorder;
