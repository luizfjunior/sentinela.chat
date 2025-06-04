
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { UserRound, Copy, ThumbsUp, ThumbsDown } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage = ({ message }: ChatMessageProps) => {
  const [displayedContent, setDisplayedContent] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showActions, setShowActions] = useState(false);
  const isUser = message.role === "user";

  // Process content to handle line breaks
  const processContent = (content: string) => {
    return content.replace(/\\n\\n/g, '\n\n').replace(/\\n/g, '\n');
  };

  const processedContent = processContent(message.content);

  // Typing animation for assistant messages
  useEffect(() => {
    if (isUser) {
      setDisplayedContent(processedContent);
      return;
    }

    if (currentIndex < processedContent.length) {
      const timeoutId = setTimeout(() => {
        setDisplayedContent(processedContent.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 15);

      return () => clearTimeout(timeoutId);
    }
  }, [currentIndex, isUser, processedContent]);

  useEffect(() => {
    if (isUser) {
      setDisplayedContent(processedContent);
    } else {
      // Reset for new assistant messages
      setCurrentIndex(0);
      setDisplayedContent("");
    }
  }, [message, isUser, processedContent]);

  const handleCopy = () => {
    navigator.clipboard.writeText(processedContent);
  };

  return (
    <div
      className={cn(
        "group relative",
        isUser ? "ml-auto max-w-[80%]" : "mr-auto max-w-full"
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div
        className={cn(
          "flex gap-4",
          isUser ? "justify-end" : "justify-start"
        )}
      >
        {!isUser && (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center flex-shrink-0">
            <span className="text-sm font-medium text-white">AI</span>
          </div>
        )}
        
        <div
          className={cn(
            "rounded-2xl px-4 py-3 max-w-full break-words",
            isUser
              ? "bg-blue-600 text-white"
              : "bg-white dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-200 dark:border-gray-700"
          )}
        >
          <p className="whitespace-pre-wrap leading-relaxed">{displayedContent}</p>
          {!isUser && currentIndex < processedContent.length && (
            <span className="ml-1 inline-block w-1 h-4 bg-blue-400 animate-pulse"/>
          )}
        </div>
        
        {isUser && (
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
            <UserRound className="h-4 w-4 text-white" />
          </div>
        )}
      </div>

      {/* Action buttons for assistant messages */}
      {!isUser && showActions && currentIndex >= processedContent.length && (
        <div className="flex items-center gap-1 mt-2 ml-12 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            onClick={handleCopy}
          >
            <Copy className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-500 hover:text-green-600 dark:text-gray-400 dark:hover:text-green-400"
          >
            <ThumbsUp className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-400"
          >
            <ThumbsDown className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
