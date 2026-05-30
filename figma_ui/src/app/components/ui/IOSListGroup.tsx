import React from 'react';
import { ChevronRight } from 'lucide-react';

export interface IOSListGroupProps {
  children: React.ReactNode;
  className?: string;
  title?: string;
}

export const IOSListGroup: React.FC<IOSListGroupProps> = ({ children, className = '', title }) => {
  return (
    <div className={`mb-6 ${className}`}>
      {title && (
        <h3 className="ml-4 mb-2 text-[13px] uppercase tracking-wide text-black/50 dark:text-white/50 font-medium">
          {title}
        </h3>
      )}
      <div className="ios-glass-panel rounded-[18px] overflow-hidden flex flex-col">
        {React.Children.map(children, (child, index) => {
          if (!React.isValidElement(child)) return null;
          const isLast = index === React.Children.count(children) - 1;
          return React.cloneElement(child, { 
            // @ts-ignore
            isLast 
          });
        })}
      </div>
    </div>
  );
};

export interface IOSListItemProps {
  icon?: React.ReactNode;
  title: string;
  subtitle?: string;
  rightContent?: React.ReactNode;
  onClick?: () => void;
  isLast?: boolean;
  destructive?: boolean;
}

export const IOSListItem: React.FC<IOSListItemProps> = ({
  icon,
  title,
  subtitle,
  rightContent,
  onClick,
  isLast = false,
  destructive = false
}) => {
  return (
    <div
      role="button"
      tabIndex={onClick ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
      className={`
        relative w-full flex items-center min-h-[56px] px-4 text-left
        ios-list-item outline-none ${onClick ? 'cursor-pointer' : 'cursor-default'}
      `}
    >
      {icon && (
        <div className="mr-3 flex items-center justify-center">
          {icon}
        </div>
      )}
      
      <div className={`
        flex-1 py-3 flex items-center justify-between
        ${!isLast ? 'border-b border-black/5 dark:border-white/10' : ''}
      `}>
        <div className="flex flex-col justify-center">
          <span className={`text-[17px] font-normal leading-tight ${destructive ? 'text-red-500' : 'text-black dark:text-white'}`}>
            {title}
          </span>
          {subtitle && (
            <span className="text-[15px] text-black/50 dark:text-white/50 mt-0.5">
              {subtitle}
            </span>
          )}
        </div>
        
        <div className="flex items-center ml-4">
          {rightContent && (
            <div className="mr-2 text-[17px] text-black/50 dark:text-white/50 flex items-center" onClick={(e) => e.stopPropagation()}>
              {rightContent}
            </div>
          )}
          {onClick && !rightContent && (
            <ChevronRight className="w-5 h-5 text-black/30 dark:text-white/30" />
          )}
        </div>
      </div>
    </div>
  );
};
