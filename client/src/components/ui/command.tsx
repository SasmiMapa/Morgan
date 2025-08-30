"use client"

import * as React from "react"
import { Command as CommandPrimitive } from "cmdk"
import { SearchIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

function Command({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive>) {
  return (
    <CommandPrimitive
      data-slot="command"
      className={cn(
        "tw-:bg-popover tw-:text-popover-foreground tw-:flex tw-:size-full tw-:flex-col tw-:overflow-hidden tw-:rounded-md",
        className
      )}
      {...props}
    />
  )
}

function CommandDialog({
  title = "Command Palette",
  description = "Search for a command to run...",
  children,
  ...props
}: React.ComponentProps<typeof Dialog> & {
  title?: string
  description?: string
}) {
  return (
    <Dialog {...props}>
      <DialogHeader className="tw-:sr-only">
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
      </DialogHeader>
      <DialogContent className="tw-:overflow-hidden tw-:p-0 tw-:sm:max-w-lg tw-:[&>button:last-child]:hidden">
        <Command className="tw-:[&_[cmdk-group-heading]]:text-muted-foreground tw-:max-h-[100svh] tw-:**:data-[slot=command-input-wrapper]:h-12 tw-:[&_[cmdk-group-heading]]:px-2 tw-:[&_[cmdk-group-heading]]:font-medium tw-:[&_[cmdk-group]]:px-2 tw-:[&_[cmdk-input]]:h-12 tw-:[&_[cmdk-item]]:px-3 tw-:[&_[cmdk-item]]:py-2">
          {children}
        </Command>
      </DialogContent>
    </Dialog>
  )
}

function CommandInput({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive.Input>) {
  return (
    <div
      className="tw-:border-input tw-:flex tw-:items-center tw-:border-b tw-:px-5"
      cmdk-input-wrapper=""
    >
      <SearchIcon size={20} className="tw-:text-muted-foreground/80 tw-:me-3" />
      <CommandPrimitive.Input
        data-slot="command-input-wrapper"
        className={cn(
          "tw-:placeholder:text-muted-foreground/70 tw-:flex tw-:h-10 tw-:w-full tw-:rounded-md tw-:bg-transparent tw-:py-3 tw-:text-sm tw-:outline-hidden tw-:disabled:cursor-not-allowed tw-:disabled:opacity-50",
          className
        )}
        {...props}
      />
    </div>
  )
}

function CommandList({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive.List>) {
  return (
    <CommandPrimitive.List
      data-slot="command-list"
      className={cn(
        "tw-:max-h-80 tw-:flex-1 tw-:overflow-x-hidden tw-:overflow-y-auto",
        className
      )}
      {...props}
    />
  )
}

function CommandEmpty({
  ...props
}: React.ComponentProps<typeof CommandPrimitive.Empty>) {
  return (
    <CommandPrimitive.Empty
      data-slot="command-empty"
      className="tw-:py-6 tw-:text-center tw-:text-sm"
      {...props}
    />
  )
}

function CommandGroup({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive.Group>) {
  return (
    <CommandPrimitive.Group
      data-slot="command-group"
      className={cn(
        "tw-:text-foreground tw-:[&_[cmdk-group-heading]]:text-muted-foreground tw-:overflow-hidden tw-:p-2 tw-:[&_[cmdk-group-heading]]:px-3 tw-:[&_[cmdk-group-heading]]:py-2 tw-:[&_[cmdk-group-heading]]:text-xs tw-:[&_[cmdk-group-heading]]:font-medium",
        className
      )}
      {...props}
    />
  )
}

function CommandSeparator({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive.Separator>) {
  return (
    <CommandPrimitive.Separator
      data-slot="command-separator"
      className={cn("tw-:bg-border tw-:-mx-1 tw-:h-px", className)}
      {...props}
    />
  )
}

function CommandItem({
  className,
  ...props
}: React.ComponentProps<typeof CommandPrimitive.Item>) {
  return (
    <CommandPrimitive.Item
      data-slot="command-item"
      className={cn(
        "tw-:data-[selected=true]:bg-accent tw-:data-[selected=true]:text-accent-foreground tw-:relative tw-:flex tw-:cursor-default tw-:items-center tw-:gap-3 tw-:rounded-md tw-:px-2 tw-:py-1.5 tw-:text-sm tw-:outline-hidden tw-:select-none tw-:data-[disabled=true]:pointer-events-none tw-:data-[disabled=true]:opacity-50 tw-:[&_svg]:pointer-events-none tw-:[&_svg]:shrink-0",
        className
      )}
      {...props}
    />
  )
}

function CommandShortcut({
  className,
  ...props
}: React.ComponentProps<"span">) {
  return (
    <kbd
      data-slot="command-shortcut"
      className={cn(
        "tw-:bg-background tw-:text-muted-foreground/70 tw-:ms-auto tw-:-me-1 tw-:inline-flex tw-:h-5 tw-:max-h-full tw-:items-center tw-:rounded tw-:border tw-:px-1 tw-:font-[inherit] tw-:text-[0.625rem] tw-:font-medium",
        className
      )}
      {...props}
    />
  )
}

export {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
}
