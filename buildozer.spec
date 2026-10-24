[app]
title = FrameSlow 4K
package.name = frameslow4k
package.domain = org.frames
source.dir = .
source.include_exts = py,png
version = 1.0
icon.filename = %(source.dir)s/icon.png
requirements = python3,kivy,numpy,opencv,pyjnius,plyer,android
orientation = all
fullscreen = 1
android.permissions = READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_VIDEO,READ_MEDIA_IMAGES
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True
p4a.branch = v2024.01.21

[buildozer]
log_level = 2
warn_on_root = 1
