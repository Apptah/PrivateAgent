# PrivateAgent Plan 7: Xcode Project + iPhone Build

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create an Xcode project that wraps the SPM package, builds an iOS app, and runs on iPhone 16 Pro with the mock engine.

**Architecture:** Minimal .xcodeproj with a single iOS app target that depends on the local SPM package. App entry point uses PrivateAgentUI.ContentView. SwiftData container configured for on-disk persistence.

**Tech Stack:** Xcode 26, iOS 18.0+, SPM local package dependency

**Depends on:** Plan 6 (UI complete)

**Produces:** An iOS app that builds and runs on iPhone 16 Pro, showing the full chat UI with mock generation.

---

### Task 1: Generate Xcode project using xcodegen

**Files:**
- Create: `project.yml` (xcodegen spec)
- Generate: `PrivateAgent.xcodeproj/`

- [ ] **Step 1: Install xcodegen if needed**

```bash
which xcodegen || brew install xcodegen
```

- [ ] **Step 2: Create project.yml**

```yaml
name: PrivateAgent
options:
  bundleIdPrefix: com.privateagent
  deploymentTarget:
    iOS: "18.0"
  xcodeVersion: "26.0"

settings:
  base:
    SWIFT_VERSION: "6.0"
    TARGETED_DEVICE_FAMILY: "1"
    GENERATE_INFOPLIST_FILE: YES

packages:
  PrivateAgent:
    path: .

targets:
  PrivateAgentApp:
    type: application
    platform: iOS
    sources:
      - path: Apps/PrivateAgentiOS
    dependencies:
      - package: PrivateAgent
        product: PrivateAgentUI
      - package: PrivateAgent
        product: FlashMoEBridge
      - package: PrivateAgent
        product: ModelHub
    settings:
      base:
        PRODUCT_BUNDLE_IDENTIFIER: com.privateagent.ios
        PRODUCT_NAME: PrivateAgent
        INFOPLIST_KEY_UILaunchScreen_Generation: YES
        INFOPLIST_KEY_CFBundleDisplayName: "PrivateAgent"
        INFOPLIST_KEY_UIApplicationSceneManifest_Generation: YES
        ASSETCATALOG_COMPILER_APPICON_NAME: AppIcon
        INFOPLIST_KEY_LSSupportsOpeningDocumentsInPlace: YES
        INFOPLIST_KEY_UIFileSharingEnabled: YES
```

- [ ] **Step 3: Generate Xcode project**

```bash
cd ~/Developer/PrivateAgent
xcodegen generate
```

- [ ] **Step 4: Update .gitignore to track the xcodeproj**

Remove `*.xcodeproj/` from .gitignore (we want this tracked since it's generated from project.yml but users may not have xcodegen).

Actually, better approach: keep xcodeproj gitignored, track project.yml. Add a note in README.

- [ ] **Step 5: Build for iOS simulator**

```bash
xcodebuild build \
    -project PrivateAgent.xcodeproj \
    -scheme PrivateAgentApp \
    -destination 'platform=iOS Simulator,name=iPhone 16 Pro' \
    -quiet 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add project.yml Apps/
git commit -m "feat: add Xcode project config for iOS app build"
```

---

### Task 2: Fix any iOS build issues

**Files:** Various — depends on what breaks

Common iOS vs macOS issues:
- `NavigationSplitView` → should be `NavigationStack` for iPhone (already done)
- Metal framework linking
- SwiftData container configuration
- Missing `#if os(iOS)` guards

- [ ] **Step 1: Build for device and fix errors**

```bash
xcodebuild build \
    -project PrivateAgent.xcodeproj \
    -scheme PrivateAgentApp \
    -destination 'generic/platform=iOS' \
    -quiet 2>&1 | tail -20
```

- [ ] **Step 2: Fix issues, rebuild, commit**

---

### Task 3: Add app icon + launch screen

**Files:**
- Create: `Apps/PrivateAgentiOS/Assets.xcassets/`

- [ ] **Step 1: Create asset catalog with placeholder icon**

Simple SF Symbol-based icon or solid color placeholder.

- [ ] **Step 2: Commit**

```bash
git add Apps/
git commit -m "feat: add app icon placeholder and asset catalog"
```

---

## Summary

After Plan 7: The app builds for iOS and can be installed on iPhone 16 Pro via Xcode. Shows the full chat UI with mock token generation.
